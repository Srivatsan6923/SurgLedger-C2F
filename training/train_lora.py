"""LoRA fine-tuning of Qwen3-VL-8B-Instruct on the FOCUS QA data.

We wrote the training loop ourselves instead of using an off-the-shelf trainer,
so that training goes through exactly the same processor call as inference.
Two settings there are easy to get wrong without any error:

  * do_sample_frames defaults to True and resamples our frames down to a handful
  * without VideoMetadata the processor assumes 24 fps, so the timestamps written
    into the prompt are wrong

Frames come from the JPEG cache built by extract_frames.py (0.5 Hz, indexed in
5 fps clip units). Frame selection uses the same sampler as inference
(decode.plan_indices_targeted), so a question is trained on the frames it will
be shown at test time, up to the 2 s spacing of the cache.

Single GPU:  python train_lora.py --parquet ... --cache ... --out ...
Multi GPU:   torchrun --standalone --nproc_per_node=4 train_lora.py ...
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference" / "resources" / "surgledger"))
from decode import CLIP_FPS, parse_timestamps, plan_indices_targeted     # noqa: E402
from format_router import DEFAULT_FO_CLASSES, route, wants_multiple_times  # noqa: E402
from pipeline import SYSTEM, FORMAT_HINT                                 # noqa: E402

CACHE_STRIDE = 10          # the cache holds every 10th clip index (0.5 Hz at 5 fps)
MIN_WINDOW_FRAMES = 4      # Qwen3-VL's video processor needs >= temporal_factor (2)
GROUP = {"1": "object_recognition", "2": "temporal_grounding", "3": "aggregation",
         "4": "event_understanding", "5": "complex_reasoning"}


def hms(v):
    h, m, s = str(v).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def build_prompt(question, fo_classes):
    """Same prompt as Pipeline._generate builds at inference."""
    fmt = route(question)
    hint = FORMAT_HINT.get(fmt, "")
    if fmt == "fo_class":
        hint += (" Choose only from these exact names: " + ", ".join(fo_classes)
                 + ". Name ONLY objects you can actually see; adding a class that "
                   "is not there makes the whole answer wrong. Most answers are "
                   "one class, or none.")
    elif fmt == "time" and wants_multiple_times(question):
        hint += " This question asks for several: comma-separated, increasing."
    # pipeline.TERSE is not appended here, because inference never adds it
    # either. Training and serving prompts have to match.
    return fmt, (question + "\n\n" + hint).strip()


class ProcedureSFT(Dataset):
    def __init__(self, rows, cache_dir, n_frames, fo_classes):
        self.rows, self.cache = rows, Path(cache_dir)
        self.n_frames, self.fo = n_frames, fo_classes
        self._avail = {}

    def __len__(self):
        return len(self.rows)

    def _frames_for(self, video):
        stem = Path(video).stem
        if stem not in self._avail:
            d = self.cache / stem
            self._avail[stem] = (d, sorted(int(p.stem) for p in d.glob("*.jpg")))
        return self._avail[stem]

    def __getitem__(self, i):
        import numpy as np
        from PIL import Image
        r = self.rows[i]
        d, avail = self._frames_for(r["video"])
        if not avail:
            raise RuntimeError("empty frame cache for " + str(r["video"]))
        # load_rows has already moved the row onto clip time, so the window is
        # [off, off + win] in cache index space and clip frame 0 is `off`.
        off, win = r["off"], r["win"]
        wav = [c for c in avail if off <= c <= off + win]
        # The video processor needs at least 2 frames and fails deep inside the
        # collate function otherwise ("t:1 must be larger than
        # temporal_factor:2"). Short windows can fall between cache frames, so
        # we widen to the nearest cached frames around the window centre.
        if len(wav) < MIN_WINDOW_FRAMES:
            mid = off + win // 2
            wav = sorted(sorted(avail, key=lambda x: abs(x - mid))[:MIN_WINDOW_FRAMES])
        win = min(win, wav[-1] - off + CACHE_STRIDE)
        # Same rule as inference (pipeline.py): time questions use the plain
        # grid, because a cited time marks where the object was, not where the
        # answer is. Everything else samples around cited timestamps.
        fmt_now = route(r["question"])
        cited = [] if fmt_now == "time" else parse_timestamps(r["question"])
        idx = plan_indices_targeted(max(win, 2), self.n_frames, cited, start_time=0.0)
        aset = set(wav)
        snapped = []
        for j in idx:
            c = int(round((off + j) / CACHE_STRIDE)) * CACHE_STRIDE   # clip -> cache
            c = min(max(c, wav[0]), wav[-1])
            if c not in aset:
                c = min(wav, key=lambda x: abs(x - c))
            snapped.append(c)
        seen, keep = set(), []
        for c in snapped:
            if c not in seen:
                seen.add(c)
                keep.append(c)
        want = min(self.n_frames, len(wav))
        # Fill up by striding over the window instead of taking frames from the
        # start. Taking them in order piled the extra frames into the first few
        # minutes of the clip, and since cap_pixels_per_frame caps the total
        # number of video tokens, those frames also lowered the resolution of
        # every other frame.
        need = want - len(keep)
        if need > 0:
            rest = [c for c in wav if c not in seen]
            if rest:
                step = max(1, len(rest) // need)
                for c in rest[::step]:
                    if len(keep) >= want:
                        break
                    seen.add(c)
                    keep.append(c)
                for c in rest:               # top up if striding came up short
                    if len(keep) >= want:
                        break
                    if c not in seen:
                        seen.add(c)
                        keep.append(c)
        keep = sorted(keep)[: self.n_frames]
        frames = [np.asarray(Image.open(d / ("%08d.jpg" % c)).convert("RGB"))
                  for c in keep]
        fmt, prompt = build_prompt(r["question"], self.fo)
        # Indices must be relative to the clip. They go into VideoMetadata, and
        # Qwen3-VL writes frame_index / fps into the prompt as timestamps. With
        # absolute cache indices, a SEGMENT clip starting at 01:57:55 would show
        # "7075.0s" while its (rebased) answer says "00:00:08".
        return {"frames": frames, "indices": [c - off for c in keep], "prompt": prompt,
                "answer": str(r["answer"]), "fmt": fmt, "win": win, "track": r["track"],
                "cache_idx": keep}


def make_collate(processor, max_len):
    from transformers.video_utils import VideoMetadata

    def collate(batch):
        b = batch[0]                                    # per-device batch size is 1
        # __getitem__ already guarantees >= MIN_WINDOW_FRAMES. This is a second
        # safety net: a 1-frame clip would raise inside a DataLoader worker and
        # take down every rank, so we skip the sample instead.
        if len(b["frames"]) < 2:
            return None
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "video"},
                                         {"type": "text", "text": b["prompt"]}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        meta = VideoMetadata(total_num_frames=b["win"], fps=CLIP_FPS,
                             duration=b["win"] / CLIP_FPS, frames_indices=b["indices"])
        tok = processor.tokenizer
        answer_text = b["answer"] + tok.eos_token
        # Tokenize prompt and answer in one processor call. Appending the answer
        # tokens afterwards breaks the token layout Qwen3-VL's get_rope_index
        # uses for its 3D position ids ("IndexError: mask [N+k] does not match
        # indexed tensor [N]").
        enc = processor(text=[text + answer_text], videos=[b["frames"]],
                        do_sample_frames=False, cap_pixels_per_frame=True,
                        video_metadata=[meta], return_tensors="pt")
        ids = enc["input_ids"]
        if ids.shape[1] > max_len:
            return None
        n_ans = len(tok(answer_text, add_special_tokens=False).input_ids)
        if n_ans >= ids.shape[1]:
            return None
        # Loss on the answer tokens only. With the prompt included, the loss
        # mostly tracks question length.
        out = dict(enc)
        out["n_ans"] = n_ans          # used by the loss, not passed to the model
        return out

    return collate


_TS_ANY = re.compile(r"\b\d{1,2}:[0-5]\d:[0-5]\d\b")
_DUR_Q = re.compile(
    r"how long|adding (together|all)|total (time|duration)|duration|cumulative", re.I)


def _hms_str(t):
    t = max(0, int(round(t)))
    return "%02d:%02d:%02d" % (t // 3600, t % 3600 // 60, t % 60)


def rebase(text, off_s):
    """Shift every hh:mm:ss in `text` back by `off_s` seconds (to clip time)."""
    if not off_s:
        return text
    return _TS_ANY.sub(lambda m: _hms_str(hms(m.group(0)) - off_s), str(text))


def track_of(path):
    """frame / segment / procedure, from the parquet path or file name.

    Works with the dataset layout (data/<track>/train.parquet) and with flattened
    copies named <dataset>_<track>_train.parquet. We fail instead of guessing,
    because treating SEGMENT rows as PROCEDURE rows would train them on the
    wrong part of the video.
    """
    s = str(path).replace("\\", "/").lower()
    for t in ("frame", "segment", "procedure"):
        if f"/{t}/" in s or f"_{t}_" in s or f"_{t}." in s:
            return t
    raise ValueError(
        f"cannot determine track from {path!r}; name it <ds>_<track>_train.parquet")


def load_rows(parquets, cache_dir, val_videos_out, val_frac, seed, oversample_cap,
              frame_ctx_s=4.0):
    import pyarrow.parquet as pq
    rows = []
    for p in parquets:
        track = track_of(p)
        t = pq.read_table(p)
        cols = {c: t.column(c).to_pylist() for c in t.column_names}
        for i in range(t.num_rows):
            r = {k: cols[k][i] for k in
                 ("video", "question", "answer", "answer_format",
                  "timestamp_start", "timestamp_end", "primary_capability")}
            # Kept for evaluation only, never used for training: the row id and
            # the ood / clinical flags let us break validation results down the
            # same way the leaderboards do. Not every parquet has them.
            for opt in ("id", "ood", "clinical_relevance", "procedure_type"):
                if opt in cols:
                    r[opt] = cols[opt][i]
            r["track"] = track
            # Row ids repeat across the two datasets, so keep the corpus too.
            low = str(p).replace("\\", "/").lower()
            r["corpus"] = "heico" if "heico" in low else (
                "lapchole" if "lapchole" in low else "unknown")
            # Window. PROCEDURE rows all start at 00:00:00, but SEGMENT and
            # FRAME rows do not, so timestamp_start has to be used.
            st, en = hms(r["timestamp_start"]), hms(r["timestamp_end"])
            if track == "frame":
                # A single moment. We give it a few seconds of context on each
                # side instead of a full frame budget.
                st, en = max(0.0, st - frame_ctx_s), en + frame_ctx_s
            off = st
            # Move the row onto clip time so a SEGMENT row looks exactly like a
            # PROCEDURE row. Timestamps in the question always shift. The answer
            # only shifts when it is a position on the timeline: a duration
            # ("for how long is a Sponge visible" -> 00:00:28) must stay as is.
            if off:
                r["question"] = rebase(r["question"], off)
                if r["answer_format"] == "time" and not _DUR_Q.search(str(r["question"])):
                    r["answer"] = rebase(r["answer"], off)
            r["off"] = int(round(off * CLIP_FPS))     # cache index of clip frame 0
            r["win"] = max(2, int(round((en - off) * CLIP_FPS)))
            rows.append(r)
    have = {p.name for p in Path(cache_dir).iterdir() if p.is_dir()}
    kept = [r for r in rows if Path(r["video"]).stem in have]
    dropped = len(rows) - len(kept)
    # Split by video. Questions about the same procedure share instruments,
    # camera and events, so a random row split would leak.
    vids = sorted({r["video"] for r in kept})
    random.Random(seed).shuffle(vids)
    val_v = set(vids[: max(1, int(len(vids) * val_frac))])
    Path(val_videos_out).write_text(json.dumps(sorted(val_v), indent=2))
    tr = [r for r in kept if r["video"] not in val_v]
    va = [r for r in kept if r["video"] in val_v]
    by = {}
    for r in tr:
        by.setdefault(GROUP[str(r["primary_capability"])[0]], []).append(r)
    counts = {g: len(v) for g, v in by.items()}
    # Balance the capability buckets. event_understanding and complex_reasoning
    # are a few percent of the rows but 40% of the score, so smaller buckets are
    # repeated up to the size of the largest one, capped at `oversample_cap`
    # repeats so they are not simply memorised.
    #
    # Note that the target is the largest bucket. With all three tracks the cap
    # is what limits the two small buckets, so adding rows to a large bucket
    # does not give them more exposure. Raising the cap does, at the cost of
    # more repetition.
    if by:
        target = max(len(v) for v in by.values())
        out = []
        for g, v in by.items():
            reps = min(oversample_cap, max(1, target // max(len(v), 1)))
            out += v * reps
        tr = out
    return tr, va, counts, dropped


def audit_alignment(rows, cache_dir, n_frames, P, n_show=10):
    """Hard checks that the frames of each row actually cover its window.

    If timestamp_start were ignored, every SEGMENT and FRAME row would train on
    the start of the video and nothing would fail. So we check at startup,
    before any GPU time is spent, and print a few samples to look at.
    """
    ds = ProcedureSFT(rows, cache_dir, n_frames, DEFAULT_FO_CLASSES)
    by_track = {}
    for i, r in enumerate(rows):
        by_track.setdefault(r["track"], []).append(i)
    P("")
    P("=== FRAME/WINDOW ALIGNMENT AUDIT ===")
    P("%-10s %-9s %-9s %-7s %-6s %s" %
      ("track", "clip_win", "cache_off", "frames", "span_s", "first gold / answer"))
    shown = 0
    for track in sorted(by_track):
        idxs = by_track[track][:: max(1, len(by_track[track]) // max(1, n_show // len(by_track)))]
        for i in idxs[: max(1, n_show // len(by_track))]:
            r, s = rows[i], ds[i]
            rel = s["indices"]
            assert rel and min(rel) >= 0, f"{track}: negative clip-relative index {min(rel)}"
            assert max(rel) <= s["win"] + CACHE_STRIDE, (
                f"{track}: index {max(rel)} outside window {s['win']}")
            assert len(s["frames"]) == len(rel), f"{track}: frame/index count mismatch"
            cov = (max(s["cache_idx"]) - min(s["cache_idx"])) / CLIP_FPS
            P("%-10s %-9d %-9d %-7d %-6.0f %s -> %r" %
              (track, s["win"], r["off"], len(rel), cov,
               r["question"][:46].replace("\n", " "), str(r["answer"])[:22]))
            shown += 1
    assert shown, "audit selected no rows"

    # Checks over the full set, not just the printed sample.
    for track, idxs in by_track.items():
        offs = [rows[i]["off"] for i in idxs]
        wins = [rows[i]["win"] for i in idxs]
        if track == "procedure":
            assert max(offs) == 0, (
                "PROCEDURE rows must start at 00:00:00; got off=%d" % max(offs))
        else:
            assert max(offs) > 0, (
                "%s rows all have off==0 -- timestamp_start was ignored, which is "
                "exactly the bug this audit exists to catch" % track)
        P("  %-10s n=%-6d off max=%-8d win median=%d"
          % (track, len(idxs), max(offs), sorted(wins)[len(wins) // 2]))

    # After rebasing, time answers have to lie inside the window, otherwise the
    # supervision points at frames the model cannot see.
    bad = 0
    for i in by_track.get("segment", [])[:2000]:
        r = rows[i]
        if r["answer_format"] != "time" or _DUR_Q.search(r["question"]):
            continue
        g = _TS_ANY.findall(str(r["answer"]))
        if len(g) != 1:
            continue
        if not (0 <= hms(g[0]) * CLIP_FPS <= r["win"]):
            bad += 1
    assert bad == 0, f"{bad} rebased SEGMENT time answers fall outside their window"
    P("=== AUDIT PASSED ===")
    P("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--cache", default="frames", help="frame cache from extract_frames.py")
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--out", default="runs/lora")
    ap.add_argument("--n-frames", type=int, default=384)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=4,
                    help="gradient accumulation; effective batch = GPUs x accum")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=65536,
                    help="384 frames is ~43k vision tokens; a 32k cap "
                         "silently skips EVERY sample")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--oversample-cap", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push-to", default="",
                    help="optional HF repo to upload each checkpoint to as it is "
                         "saved (useful on machines with temporary disks). Needs "
                         "a write-scoped HF_TOKEN.")
    ap.add_argument("--init-adapter", default="",
                    help="start from an existing LoRA adapter instead of a new "
                         "one. The adapter directory is only read.")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = derive from epochs")
    ap.add_argument("--deadline-min", type=float, default=0.0,
                    help="stop and save once this many minutes have elapsed (0=off)")
    a = ap.parse_args()

    # Check before loading the model: 384 frames need ~43k tokens, and with a
    # smaller --max-len every sample would be skipped without an error.
    est_tokens = a.n_frames * 112 + 512
    if est_tokens > a.max_len:
        raise SystemExit("--max-len %d is below the ~%d tokens %d frames needs; "
                         "every sample would be skipped"
                         % (a.max_len, est_tokens, a.n_frames))

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    dev = "cuda:%d" % local

    def P(*m):
        if rank == 0:
            print(*m, flush=True)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(a.model)
    tr, va, counts, dropped = load_rows(
        a.parquet, a.cache, Path(a.out) / "val_videos.json",
        a.val_frac, a.seed, a.oversample_cap)
    P("train rows (after oversample) = %d   val rows = %d   dropped (no cache) = %d"
      % (len(tr), len(va), dropped))
    P("bucket counts before oversample: %s" % counts)
    audit_alignment(tr, a.cache, a.n_frames, P)

    # Pass both `dtype` (transformers 5.x) and `torch_dtype` (older versions).
    # With only one of them, one version ignores it and loads the model in fp32
    # (~35 GB), which looks like an out-of-memory problem with long sequences.
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            a.model, dtype=torch.bfloat16, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to(dev)
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    # Stop rather than train at half speed in twice the memory.
    dts = {p.dtype for p in model.parameters()}
    got = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    P("model dtypes=%s  weights=%.1f GB" % (sorted(str(d) for d in dts), got))
    if torch.bfloat16 not in dts or got > 24:
        raise SystemExit("MODEL DID NOT LOAD IN bf16 (%.1f GB) -- refusing to train" % got)
    model.config.use_cache = False
    # The vision tower stays frozen. It also saves the memory we need for long
    # frame sequences.
    n_frozen = 0
    for n, p in model.named_parameters():
        if "visual" in n or "vision" in n:
            p.requires_grad_(False)
            n_frozen += 1
    if a.init_adapter:
        # Continue from an existing adapter (we started the final run from an
        # early checkpoint of a PROCEDURE-only run). is_trainable=True is
        # required, otherwise the adapter is loaded frozen and nothing trains.
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_tr > 0, (
            "warm-started adapter has 0 trainable params -- is_trainable was ignored")
        P("warm start from %s (%.1f M trainable)" % (a.init_adapter, n_tr / 1e6))
    else:
        lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lcfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # model.train() is required: transformers only applies gradient
    # checkpointing when the module is in training mode, and from_pretrained
    # returns the model in eval mode. Without it activations are kept for every
    # layer and the forward pass runs out of memory.
    model.train()
    gc_on = [m for m in model.modules() if getattr(m, "gradient_checkpointing", False)]
    P("training mode=%s  layers with checkpointing=%d" % (model.training, len(gc_on)))
    if not model.training or not gc_on:
        raise SystemExit("gradient checkpointing NOT active -- refusing to train")
    if rank == 0:
        model.print_trainable_parameters()
        P("frozen vision params: %d" % n_frozen)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=True)

    ds = ProcedureSFT(tr, a.cache, a.n_frames, DEFAULT_FO_CLASSES)
    # Group samples by length. With batch size 1 per GPU, every optimizer step
    # waits for the slowest rank, and FRAME rows (a few frames) mixed with
    # PROCEDURE rows (384 frames) would make almost every step as slow as a
    # PROCEDURE step. DistributedLengthGroupedSampler gives the ranks of one
    # step samples of similar length.
    sampler = None
    if ddp:
        lengths = [r["win"] for r in tr]     # window length as a proxy for tokens
        try:
            from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
            sampler = DistributedLengthGroupedSampler(
                batch_size=1, dataset=ds, lengths=lengths, seed=a.seed)
            P("sampler: DistributedLengthGroupedSampler over %d rows" % len(lengths))
        except Exception as e:               # say so if we have to fall back
            P("!! DistributedLengthGroupedSampler unavailable (%s)" % e)
            P("!! falling back to DistributedSampler -- expect up to 3x the step time")
            sampler = DistributedSampler(ds, shuffle=True, seed=a.seed)
    dl = DataLoader(ds, batch_size=1, sampler=sampler, shuffle=(sampler is None),
                    num_workers=4, collate_fn=make_collate(processor, a.max_len),
                    pin_memory=True, drop_last=True)

    world = int(os.environ.get("WORLD_SIZE", 1))
    steps = a.max_steps or max(1, int(len(dl) * a.epochs / a.accum))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    warm = max(1, int(0.03 * steps))

    def lr_at(s):
        if s < warm:
            return s / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    P("world=%d steps=%d accum=%d effective_batch=%d n_frames=%d ~tokens/sample=%d lr=%s"
      % (world, steps, a.accum, world * a.accum, a.n_frames, est_tokens, a.lr))

    t0 = time.time()
    step = skipped = ep = seen = 0
    running = 0.0
    stop = False
    while not stop:
        if sampler:
            sampler.set_epoch(ep)
        for micro, batch in enumerate(dl):
            if batch is None:                 # over max_len
                skipped += 1
                if skipped in (1, 10, 100) or skipped % 500 == 0:
                    P("WARNING: %d samples skipped for exceeding --max-len %d "
                      "-- if this keeps climbing the cap is wrong for n_frames=%d"
                      % (skipped, a.max_len, a.n_frames))
                continue
            n_ans = int(batch.pop("n_ans"))
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            ids = batch["input_ids"]
            out = model(**batch, logits_to_keep=n_ans + 1)
            # The logits cover the last n_ans+1 positions; position -k predicts
            # token -k+1, so drop the last logit and score the last n_ans tokens.
            lg = out.logits[:, :-1, :].float()
            loss = torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.size(-1)), ids[:, -n_ans:].reshape(-1))
            (loss / a.accum).backward()
            running += loss.item()
            seen += 1
            if (micro + 1) % a.accum:
                continue
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            # Probe mode (--max-steps 1 or 2): print loss, sequence length and
            # memory per rank, to check that a configuration fits before
            # starting the real run.
            if a.max_steps and a.max_steps <= 2:
                P("PROBE step=%d rank=%d/%d loss=%.4f grad_norm=%.4f "
                  "seq_len=%d supervised_tokens=%d frames=%d "
                  "vram_alloc=%.1fGB vram_peak=%.1fGB sec/step=%.2f"
                  % (step, rank, world, loss.item(), float(gnorm),
                     ids.shape[1], n_ans, len(batch.get("pixel_values_videos", [[]])[0])
                     if "pixel_values_videos" in batch else -1,
                     torch.cuda.memory_allocated(dev) / 1e9,
                     torch.cuda.max_memory_allocated(dev) / 1e9,
                     (time.time() - t0) / step))
            if step % 10 == 0:
                el = (time.time() - t0) / 60
                P("step %d/%d loss=%.4f lr=%.2e %.1fmin eta=%.1fmin skipped=%d"
                  % (step, steps, running / max(seen, 1), sched.get_last_lr()[0],
                     el, el / step * (steps - step), skipped))
                running = 0.0
                seen = 0
            if rank == 0 and step % a.save_every == 0:
                d = Path(a.out) / ("ckpt-%d" % step)
                (model.module if ddp else model).save_pretrained(d)
                P("saved %s" % d)
                # Optionally upload each checkpoint as well. Upload errors are
                # logged but do not stop training.
                if a.push_to:
                    try:
                        from huggingface_hub import HfApi
                        t1 = time.time()
                        HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(
                            folder_path=str(d), repo_id=a.push_to,
                            path_in_repo=d.name, repo_type="model",
                            commit_message="ckpt-%d" % step)
                        P("mirrored %s -> %s (%.0f s)" % (d.name, a.push_to,
                                                          time.time() - t1))
                    except Exception as e:
                        P("!! mirror of %s FAILED (%s) -- training continues, but this "
                          "checkpoint exists only on ephemeral disk" % (d.name, e))
            if step >= steps:
                stop = True
                break
            if a.deadline_min and (time.time() - t0) / 60 > a.deadline_min:
                P("DEADLINE %.0f min reached at step %d" % (a.deadline_min, step))
                stop = True
                break
        ep += 1
        if ep > 50:
            break
    if rank == 0 and a.max_steps and a.max_steps <= 2:
        # A probe should not change anything on disk. Use a large --save-every
        # so no checkpoint is written; the summary below lists what was.
        P("")
        P("=== PROBE EPILOGUE ===")
        P("world=%d effective_batch=%d n_frames=%d max_len=%d"
          % (world, world * a.accum, a.n_frames, a.max_len))
        P("samples skipped for exceeding max_len: %d of %d seen  (truncation: NONE -- "
          "over-length samples are DROPPED, never truncated)" % (skipped, skipped + seen))
        init = getattr(a, "init_adapter", None)
        if init and Path(init).exists():
            import hashlib
            h = hashlib.sha256()
            for f in sorted(Path(init).rglob("*")):
                if f.is_file():
                    h.update(f.read_bytes())
            P("init adapter %s sha256=%s  (read-only, never written)"
              % (init, h.hexdigest()[:16]))
        outp = Path(a.out)
        wrote = sorted(p.relative_to(outp).as_posix()
                       for p in outp.rglob("*") if p.is_file()) if outp.exists() else []
        P("files written under --out: %s" % (wrote or "NONE"))
        P("checkpoints written: %d (save_every=%d, so a %d-step probe writes none)"
          % (len([w for w in wrote if "ckpt-" in w]), a.save_every, a.max_steps))
        P("=== PROBE COMPLETE -- stopping before any real training ===")
        if ddp:
            dist.destroy_process_group()
        return

    if rank == 0:
        d = Path(a.out) / "final"
        (model.module if ddp else model).save_pretrained(d)
        json.dump({"steps": step, "skipped": skipped, "n_frames": a.n_frames,
                   "minutes": (time.time() - t0) / 60},
                  open(Path(a.out) / "train_summary.json", "w"), indent=2)
        P("DONE steps=%d skipped=%d minutes=%.1f -> %s"
          % (step, skipped, (time.time() - t0) / 60, d))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
