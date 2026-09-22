"""Request -> sampled frames -> Qwen3-VL -> answer in the expected format.

Two practical details worth knowing:

* We run a dummy forward pass during setup. Otherwise the first question pays
  for CUDA and decord initialisation (around 30 s in our tests) inside its own
  time budget instead of the 120 s setup allowance.
* The platform drops questions once `120 + B*30` s is exceeded, and a 20%
  overrun loses the whole batch, so the frame count is reduced as the deadline
  gets close (see decode.frames_for_budget).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from decode import (CLIP_FPS, budget_for, frame_to_original_time, parse_timestamps,
                    plan_indices, plan_indices_targeted)
from format_router import (DEFAULT_FO_CLASSES, FALLBACK, _seconds_to_hms,
                           route, serialize, wants_multiple_times)
from template_priors import judge_prior_for, prior_for

log = logging.getLogger(__name__)

# Frames vs. resolution: on a video-disjoint held-out split, going from 128 to
# 384 frames helped, while going from 448 to 672 px width made no difference.
# So the budget goes into temporal coverage and we keep the cheaper width.
# After fine-tuning the model got much better at temporal questions and frame
# density became the limiting factor (at 192 frames a typical clip is one frame
# every ~30 s, while FO events last seconds), so the final setting is 384.
# The vision tower was frozen during fine-tuning, so per-frame features are the
# same as the base model; only the sequence length changes. 384 frames take
# roughly 10-12 s per question on an L40S, well within the 30 s budget.
MAX_FRAMES = 384
FRAME_WIDTH = 448  # height follows the clip's aspect ratio, rounded to /32
MAX_NEW_TOKENS = 48
# OpenEnded.verify allows 300 characters and some reference answers use most of
# that. 48 tokens cuts off around 190 characters, which the judge reads as an
# incomplete answer, so the two judge-graded formats get a longer limit.
# Generation is cheap next to the 384-frame prefill.
MAX_NEW_TOKENS_OPEN = 110
# Threshold on log P(yes) - log P(no) for binary questions. A positive value
# makes "no" the default, which would match the slight majority of "no"
# answers in the data. 0.0 is plain argmax, which is what we shipped.
BINARY_TAU = 0.0
_RESIZE_CHUNK = 64   # keeps the float32 resize buffer around 0.45 GB

# Let decord resize while decoding instead of decoding at full resolution and
# resizing in torch. It is faster and uses much less memory, but the resampling
# filter differs, so every frame changes slightly. We kept the torch path in
# the submission.
DECODE_SIDE_RESIZE = False

# Formats where the per-template prior (template_priors.PRIORS) is used instead
# of the model. For the zero-shot model this helped on fo_class and number, but
# after LoRA fine-tuning the model beat the priors on both, so the set is empty.
# Only worth restoring if you run the base model without the adapter.
DEFER_FORMATS = frozenset()

# Short answers are only enforced for the regex-parsed formats. open_ended
# questions are graded by an LLM judge that checks semantic equivalence and
# rejects answers that are incomplete, and the reference answers there are
# often full sentences, so a one-word reply would lose points.
SYSTEM = (
    "You are a surgical video analyst. You are shown frames sampled across an "
    "endoscopic procedure, in chronological order. Answer using the visual evidence "
    "together with standard surgical knowledge of the procedure."
)
TERSE = " Reply with the answer value ALONE - no explanation, no units, no full sentences."

# The fine-tuning targets use hh:mm:ss, so we ask for hh:mm:ss at inference
# too. An earlier version asked for seconds, because the base model could not
# convert to hh:mm:ss on its own; the "seconds" option is kept for that case.
# Either way serialize() normalises the answer to hh:mm:ss, and questions
# asking for several timestamps get the plural hint in _generate.
TIME_HINT_STYLE = "hms"          # "hms" matches the fine-tuning targets
_TIME_HINT = {
    "hms": ("Answer with the time as a single hh:mm:ss timestamp (e.g. 01:04:07). "
            "Give exactly one timestamp."),
    "seconds": ("Answer with the time in SECONDS from the start of the video, as a "
                "single plain number (e.g. 3847). Give exactly one number."),
}

# Output instruction per format. The question already states the format, but
# repeating it as a hard constraint cuts down on extra prose.
FORMAT_HINT = {
    "binary": "Answer exactly one word: yes or no.",
    "number": "Answer with a single non-negative integer, digits only.",
    "time": _TIME_HINT[TIME_HINT_STYLE],
    "fo_class": "Answer with foreign-object class name(s), comma-separated, or none.",
    "percentage": "Answer with a single number, no % sign.",
    "multiple_choice": "Answer with one of: top/left, top/right, bottom/left, bottom/right.",
    # These are judged semantically and usually describe their own answer
    # format, so we only ask for a complete answer within the 300-character
    # limit of OpenEnded.verify().
    "open_ended": ("Answer directly and completely, following any format the question "
                   "itself specifies. At most 300 characters."),
}


def _resize_hw(h: int, w: int, width: int = FRAME_WIDTH) -> tuple[int, int]:
    """Scale to `width` wide, both sides a multiple of 32 (patch 16 x merge 2)."""
    scale = width / max(w, 1)
    nh = max(32, int(round(h * scale / 32)) * 32)
    nw = max(32, int(round(w * scale / 32)) * 32)
    return nh, nw


def _shift_hms(answer: str, offset_s: float) -> str:
    """Move a time answer from clip time onto the procedure timeline.

    All PROCEDURE training clips start at 00:00:00, so the model answers in clip
    time. If a clip starts later (as SEGMENT windows do), the reference answer
    is on the original timeline, so we add start_time here. This is done in
    code rather than in the prompt, because the model never saw a non-zero
    offset during training.
    """
    off = int(offset_s)
    if off <= 0:
        return answer
    out = []
    for part in answer.split(","):
        try:
            h, m, sec = (int(x) for x in part.strip().split(":"))
        except ValueError:
            return answer          # a prior or malformed answer, leave it alone
        out.append(_seconds_to_hms(h * 3600 + m * 60 + sec + off))
    return ", ".join(out)


class Pipeline:
    def __init__(self, model_dir: str, device: str = "cuda",
                 adapter_dir: str | None = None):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device
        # bfloat16 is only native from sm_80 (Ampere) onwards. On older GPUs
        # torch emulates it and inference becomes dramatically slower, which
        # would blow the time budget. Both evaluation GPUs support it, but we
        # still pick the dtype from the hardware.
        dtype = torch.bfloat16
        if device == "cuda" and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 8:
                dtype = torch.float16
                log.warning("compute capability sm_%d0 has no native bf16; using fp16", major)
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=dtype,
            device_map=device,
            attn_implementation="sdpa",
        ).eval()
        # Merge the adapter into the weights instead of running it as a live
        # PEFT adapter. The live version (W.x + B(A.x)) gave slightly different
        # greedy outputs than the merged (W+BA).x on a small fraction of
        # questions, while the merged model reproduced our validation run exactly.
        if adapter_dir:
            from peft import PeftModel
            log.info("merging adapter %s", adapter_dir)
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_dir)).merge_and_unload().eval()
        self.fo_classes = DEFAULT_FO_CLASSES
        self._yn_cache = None
        # Decision threshold for binary questions, see BINARY_TAU.
        self.binary_tau = BINARY_TAU
        self.last_binary_margin = None
        self.last_raw = None

    def set_fo_classes(self, definitions_text: str) -> None:
        """Use the FO classes listed in /input/FO_definitions.json.

        The test set may contain classes beyond the documented ten, so the list
        is read at runtime instead of hardcoded.
        """
        found = [c for c in DEFAULT_FO_CLASSES if c.lower() in definitions_text.lower()]
        # In the file a class name is a line underlined with dashes, while
        # section titles are underlined with equals signs:
        #     Foreign Object Classes      <- section  (===)
        #     ======================
        #     Sponge                      <- class    (---)
        #     ------
        # A looser rule picked up the section title as an extra class, and an
        # unknown class name makes FOClass.read() fail.
        lines = definitions_text.splitlines()
        extra = []
        for i, line in enumerate(lines[:-1]):
            name, rule = line.strip(), lines[i + 1].strip()
            if not name or len(rule) < 3 or set(rule) != {"-"}:
                continue
            if name not in found and 2 < len(name) < 40:
                extra.append(name)
        self.fo_classes = tuple(dict.fromkeys([*found, *extra])) or DEFAULT_FO_CLASSES
        log.info("FO classes in play (%d): %s", len(self.fo_classes), list(self.fo_classes))

    def warmup(self) -> None:
        """Pay the CUDA/kernel init cost here, inside the setup allowance."""
        dummy = [np.zeros((64, 64, 3), dtype=np.uint8)] * 4
        try:
            # Warm up both paths: generate() for most formats and the single
            # forward pass used for binary questions. They use different
            # kernels.
            self._generate(dummy, "Warm-up. Answer: no.", "number")
            self._generate(dummy, "Warm-up. Answer: no.", "binary", binary_margin=True)
            log.info("Warm-up forward pass complete")
        except Exception:
            log.exception("Warm-up failed (continuing; first question will pay the cost)")

    @property
    def _yes_ids(self):
        if getattr(self, "_yn_cache", None) is None:
            tok = self.processor.tokenizer
            def ids(words):
                out = set()
                for w in words:
                    enc = tok.encode(w, add_special_tokens=False)
                    if len(enc) == 1:
                        out.add(enc[0])
                return sorted(out) or [tok.encode(words[0], add_special_tokens=False)[0]]
            self._yn_cache = (ids(["yes", "Yes", " yes", " Yes", "YES"]),
                              ids(["no", "No", " no", " No", "NO"]))
            log.info("binary decision tokens: %d yes / %d no",
                     len(self._yn_cache[0]), len(self._yn_cache[1]))
        return self._yn_cache[0]

    @property
    def _no_ids(self):
        self._yes_ids  # populate the cache
        return self._yn_cache[1]

    def _generate(self, frames: list[np.ndarray], question: str, fmt: str,
                  metadata=None,
                  binary_margin: bool = False):
        hint = FORMAT_HINT.get(fmt, "")
        if fmt == "fo_class":
            # fo_class is scored by exact set equality against a fixed list of
            # class names. Without the list the model makes up names ("gauze");
            # with only the list it tends to recite all of it. Every extra name
            # makes the answer wrong, so the list comes with an explicit
            # instruction to name only what is visible.
            hint += (" Choose only from these exact names: " + ", ".join(self.fo_classes)
                     + ". Name ONLY objects you can actually see; adding a class that "
                       "is not there makes the whole answer wrong. Most answers are "
                       "one class, or none.")
        elif fmt == "time" and wants_multiple_times(question):
            hint += " This question asks for several: comma-separated, increasing."
        prompt = f"{question}\n\n{hint}".strip()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # do_sample_frames=False is important. It defaults to True, and without
        # video_metadata the processor assumes 24 fps and resamples to 2 fps,
        # which quietly turns 64 frames into about 5. We already chose the
        # frames, so the processor should use them as they are.
        # cap_pixels_per_frame matches the token budget of the qwen-vl-utils
        # reference instead of the much larger default.
        # video_metadata is needed as well: Qwen3-VL writes a timestamp for each
        # frame into the prompt, computed as frame_index / fps. Without it the
        # model assumes 24 fps and every timestamp is off by about 5x. Passing
        # the real frame indices at 5 fps makes them exact.
        kwargs = dict(text=[text], videos=[frames], do_sample_frames=False,
                      cap_pixels_per_frame=True, return_tensors="pt")
        if metadata is not None:
            kwargs["video_metadata"] = [metadata]
        try:
            inputs = self.processor(**kwargs).to(self.device)
        except (TypeError, ValueError):
            # If a processor version does not accept one of the optional
            # arguments, retry without them. Worse timestamps are better than
            # falling back to the prior on every question.
            log.warning("processor rejected optional kwargs; retrying minimally",
                        exc_info=True)
            inputs = self.processor(text=[text], videos=[frames],
                                    do_sample_frames=False,
                                    return_tensors="pt").to(self.device)
        if log.isEnabledFor(logging.DEBUG):
            pv = inputs.get("pixel_values_videos")
            log.debug("visual grid=%s tokens=%d", tuple(pv.shape) if pv is not None else None,
                      int(inputs["input_ids"].shape[1]))

        if binary_margin:
            # Binary questions: one forward pass and compare the "yes" and
            # "no" logits at the first answer position, instead of free
            # generation. This keeps the model's ranking and lets us choose the
            # threshold (BINARY_TAU) separately.
            # logits_to_keep=1 runs the LM head on the last position only.
            # Without it the head produces a [1, seq_len, vocab] tensor, which
            # for a 384-frame clip is a multi-GB allocation and can run out of
            # memory. generate() sets this internally; a direct call has to.
            with torch.inference_mode():
                logits = self.model(**inputs, logits_to_keep=1).logits[0, -1, :].float()
            return float(logits[self._yes_ids].max() - logits[self._no_ids].max())

        with torch.inference_mode():
            budget = (MAX_NEW_TOKENS_OPEN if fmt in ("open_ended", "multiple_choice")
                      else MAX_NEW_TOKENS)
            out = self.model.generate(
                **inputs, max_new_tokens=budget, do_sample=False
            )
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        # Kept for logging and offline analysis only; nothing downstream reads it.
        self.last_raw = self.processor.decode(trimmed, skip_special_tokens=True).strip()
        return self.last_raw

    def answer(self, req, clip_path, max_frames: int = MAX_FRAMES,
               width: int = FRAME_WIDTH, clip_frames: int | None = None,
               frames=None, frame_indices=None) -> str:
        """Answer one question, returned in the canonical format.

        `clip_frames` limits how much of `clip_path` counts as the question's
        clip. On the platform it is None, because each clip is already cut to
        [start_time, end_time]. It is used by the re-read in inference.py and by
        our offline evaluation, which reuses one full-length video for all of
        its questions.
        """
        import decord  # after torch

        # `frames` / `frame_indices` let a caller pass frames that are already
        # decoded. The re-read uses this, and so did our offline evaluation
        # (which reads from a frame cache instead of re-decoding every video).

        fmt = route(req.question)
        # Reset for every question, so a failure can never leave the previous
        # question's margin behind.
        self.last_binary_margin = None

        # Answer from a template prior before touching the video, if one
        # exists for this template (see DEFER_FORMATS and template_priors).
        # DEFER_FORMATS applies to whole formats, JUDGE_PRIORS to individual
        # judge-graded templates whose answer is almost always the same.
        p = prior_for(req.question, fmt) if fmt in DEFER_FORMATS else None
        if p is None:
            p = judge_prior_for(req.question, fmt)
        if p is not None:
            log.debug("%s fmt=%s deferred to template prior %r (rate %.3f, n=%d)",
                      req.qID, fmt, p[0], p[1], p[2])
            return serialize(p[0], fmt, self.fo_classes, question=req.question)
        try:
            if frames is not None:
                n = int(clip_frames or (max(frame_indices) + 1 if frame_indices else len(frames)))
                idx = list(frame_indices or range(len(frames)))
                batch = np.asarray(frames)
                vr = None
            else:
                if DECODE_SIDE_RESIZE:
                    # decord needs the native size first, so open once to read
                    # it, then reopen with the target size.
                    probe = decord.VideoReader(str(clip_path), ctx=decord.cpu(0),
                                               num_threads=1)
                    ph, pw = probe[0].shape[:2]
                    del probe
                    th, tw = _resize_hw(ph, pw, width)
                    vr = decord.VideoReader(str(clip_path), ctx=decord.cpu(0),
                                            num_threads=2, width=tw, height=th)
                else:
                    vr = decord.VideoReader(str(clip_path), ctx=decord.cpu(0), num_threads=2)
                n = len(vr)
                if clip_frames is not None and 0 < clip_frames < n:
                    n = int(clip_frames)
            # Sample around the times a question mentions, but only when that
            # time is the evidence. For time questions it usually is not: in
            # "the Sponge at 00:09:19, when is it retrieved?" the cited time is
            # where the object was, and the answer is somewhere else. Targeted
            # sampling hurt recall for time questions in our tests, and it is
            # slower because the windows are off the keyframe grid, so time
            # questions use the plain grid.
            # Binary and fo_class questions are the opposite case: they often
            # ask about the state at the cited time ("does the Clip at T1 also
            # appear at T2"), so they keep targeted sampling.
            if vr is not None:
                cited = parse_timestamps(req.question)
                if fmt == "time":
                    idx = plan_indices(n, max_frames)
                else:
                    idx = plan_indices_targeted(
                        n, max_frames, cited,
                        start_time=getattr(req, "start_time", 0.0) or 0.0)
                batch = vr.get_batch(idx).asnumpy()  # (T, H, W, 3) uint8
                del vr

            h, w = batch.shape[1], batch.shape[2]
            nh, nw = _resize_hw(h, w, width)
            if (h, w) == (nh, nw):
                # Already at the target size (DECODE_SIDE_RESIZE), nothing to do.
                frames = list(batch)
            else:
                # Resize in chunks. Converting the whole batch to float32 at the
                # native 576x1024 takes several GB for long clips; chunking keeps
                # the buffer to _RESIZE_CHUNK frames.
                frames = []
                for s in range(0, batch.shape[0], _RESIZE_CHUNK):
                    c = torch.from_numpy(batch[s:s + _RESIZE_CHUNK]).permute(0, 3, 1, 2).float()
                    c = torch.nn.functional.interpolate(c, size=(nh, nw), mode="bilinear",
                                                        align_corners=False)
                    frames.extend(c.permute(0, 2, 3, 1).clamp(0, 255).byte().numpy())
                    del c
            del batch

            meta = None
            try:
                from transformers.video_utils import VideoMetadata
                meta = VideoMetadata(total_num_frames=n, fps=CLIP_FPS,
                                     duration=n / CLIP_FPS, frames_indices=idx)
            except Exception:
                log.warning("VideoMetadata unavailable; frame timestamps will be wrong")

            offset = getattr(req, "start_time", 0.0) or 0.0
            if fmt == "binary":
                # Decide from the yes/no margin and the threshold instead of
                # free text. A higher binary_tau answers "no" more often.
                m = self._generate(frames, req.question, fmt, metadata=meta,
                                   binary_margin=True)
                self.last_binary_margin = m
                ans = "yes" if m > self.binary_tau else "no"
                log.debug("%s binary margin=%.3f tau=%.3f -> %s",
                          req.qID, m, self.binary_tau, ans)
                return ans

            raw = self._generate(frames, req.question, fmt, metadata=meta)
            log.debug("%s fmt=%s frames=%d raw=%r", req.qID, fmt, len(idx), raw)
            ans = serialize(raw, fmt, self.fo_classes, question=req.question)
            return _shift_hms(ans, offset) if fmt == "time" else ans
        except Exception:
            log.exception("%s failed; emitting %s prior", req.qID, fmt)
            return FALLBACK[fmt]


def demo() -> None:
    """Checks the pure logic; the model path needs a GPU and is tested in the container."""
    assert _resize_hw(576, 1024) == (256, 448)
    assert _resize_hw(360, 640) == (256, 448)
    assert all(v % 32 == 0 for v in _resize_hw(576, 720))
    assert route("Please answer with yes or no.") == "binary"
    assert serialize("maybe 3 things", "number") == "3"
    assert frame_to_original_time(50, 10.0) == 20.0
    assert plan_indices(88900, 64)[0] == 0
    # A cited timestamp has to change which frames we look at.
    q = "Does the Sponge at 00:50:00 also appear at 01:00:00? Please answer with yes or no."
    cited = parse_timestamps(q)
    assert cited == [3000.0, 3600.0], cited
    uni = plan_indices(29945, 128)
    tgt = plan_indices_targeted(29945, 128, cited)
    assert tgt != uni, "targeted sampling collapsed to uniform"
    assert sum(abs(i - 15000) <= 225 for i in tgt) > sum(abs(i - 15000) <= 225 for i in uni)
    # That question is binary. A time question citing the same timestamp must
    # not be targeted, since the cited time is not where the answer is.
    qt = "The Sponge at 00:50:00 -- when is it retrieved? Please answer in hh:mm:ss."
    assert route(qt) == "time", route(qt)
    assert route(q) == "binary", route(q)
    assert parse_timestamps(qt) == [3000.0]          # it does cite one
    # The switch is on fmt, so a time question uses the uniform grid.
    assert plan_indices(29945, 128) == uni
    # Clips that start mid-procedure: answers move onto the procedure timeline.
    assert _shift_hms("00:00:57", 555.0) == "00:10:12"
    assert _shift_hms("00:00:10, 00:01:00", 3600.0) == "01:00:10, 01:01:00"
    assert _shift_hms("00:12:34", 0.0) == "00:12:34"      # clip starts at 0: unchanged
    assert _shift_hms("none", 555.0) == "none"            # a non-time prior is left alone
    assert budget_for(20) == 540.0
    assert CLIP_FPS == 5.0
    # clip_frames has to shrink the sampled range, otherwise the full video
    # would be used.
    assert max(plan_indices_targeted(29945, 128, [])) > 20000
    assert max(plan_indices_targeted(7000, 128, [])) <= 7000
    q_fo = ("What types of foreign objects are seen between 00:40:00 and 01:02:01? "
            "Please provide the class name(s) or answer with none.")
    assert route(q_fo) == "fo_class"
    assert "open_ended" not in DEFER_FORMATS and "time" not in DEFER_FORMATS
    assert serialize("none", "open_ended") == "none"
    assert serialize("1.", "number") == "1"
    assert serialize("bottom/left", "multiple_choice") == "bottom/left"
    print("pipeline self-check OK")


if __name__ == "__main__":
    demo()
