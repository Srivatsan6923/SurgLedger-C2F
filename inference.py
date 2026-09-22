"""Entry point for the ORena FOCUS PROCEDURE track container.

The batch loop, per-question error handling and answer.json writing follow the
organizers' template. What changes is how each question is answered:

  1. route the question to one of the seven answer formats,
  2. sample frames on the 5 s keyframe grid (384 frames, 768 for time questions),
  3. answer with Qwen3-VL-8B plus our merged LoRA,
  4. for time questions, re-read a +/-120 s window around the first answer
     and ask the same model again (coarse-to-fine re-read).

Everything is wrapped so a failing question emits its format prior instead of
taking down the batch.
"""

import json
import logging
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch

# decord has to be imported after torch, otherwise CUDA init can break.
import decord  # noqa: F401
from focus import Request, Response, load_requests, save_items

RESOURCES_PATH = Path(__file__).parent / "resources"
sys.path.insert(0, str(RESOURCES_PATH / "surgledger"))

from decode import budget_for, frames_for_budget  # noqa: E402
from format_router import FALLBACK, route  # noqa: E402
from pipeline import MAX_FRAMES, Pipeline  # noqa: E402

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_DIR = RESOURCES_PATH / "model"
# If a LoRA adapter is baked into the image it is merged at startup. Without
# one the base model runs as is, so the image still works.
ADAPTER_DIR = RESOURCES_PATH / "adapter"


class ClipSource:
    """Map a qID to a video file we can decode.

    On the platform the clips arrive as a single /input/batch-videos.zip, while
    the template's local test data uses plain/ and overlayed/ folders. We support
    both (folders first, then the zip). If only the folders were supported, every
    clip would be missing on the platform and the batch would quietly fall back
    to priors.

    Zip members are extracted one at a time inside each question's budget. A
    batch can be tens of GB, which does not fit in the setup allowance or on
    scratch disk.
    """

    def __init__(self, input_path: Path, workdir: Path):
        self.dir: Path | None = None
        self.zf: zipfile.ZipFile | None = None
        self.index: dict[str, str] = {}
        self.workdir = workdir

        for name in ("plain", "overlayed"):
            d = input_path / name
            if d.is_dir() and any(d.glob("*.mp4")):
                self.dir = d
                log.info("Clips: directory %s", d)
                return

        zpath = input_path / "batch-videos.zip"
        if not zpath.exists():
            log.error("No clips: neither %s/plain nor %s", input_path, zpath)
            return
        self.zf = zipfile.ZipFile(zpath)
        for m in self.zf.namelist():
            if not m.lower().endswith(".mp4"):
                continue
            qid = Path(m).stem
            # Prefer the plain clip. The overlayed version only adds a burned-in
            # clock, and we compute time from frame indices anyway.
            if qid not in self.index or "plain" in m.lower():
                self.index[qid] = m
        log.info("Clips: zip %s -> %d mp4(s)", zpath, len(self.index))

    def get(self, qid: str) -> Path | None:
        if self.dir is not None:
            p = self.dir / f"{qid}.mp4"
            return p if p.exists() else None
        member = self.index.get(str(qid))
        if member is None:
            log.error("qID %s not found in batch-videos.zip", qid)
            return None
        dst = self.workdir / f"{qid}.mp4"
        if not dst.exists():
            t0 = time.monotonic()
            with self.zf.open(member) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out, 4 * 1024 * 1024)
            log.info("  extracted %s (%.2f GB) in %.1fs",
                     member, dst.stat().st_size / 1e9, time.monotonic() - t0)
        return dst

    def release(self, qid: str) -> None:
        """Delete the extracted clip; the whole batch will not fit on disk."""
        if self.zf is None:
            return
        try:
            (self.workdir / f"{qid}.mp4").unlink(missing_ok=True)
        except OSError:
            log.warning("could not remove scratch clip for %s", qid, exc_info=True)

    def close(self) -> None:
        if self.zf is not None:
            self.zf.close()
            self.zf = None

# Once the remaining budget per unanswered question drops below this, we start
# using fewer frames. Answering with less context is better than overrunning.
DEGRADE_AT_S = 20.0
MIN_FRAMES = 8

# Time questions get twice the frames. A "when did X happen" answer is a single
# moment, and 384 frames over a 4.5 h procedure is one frame every ~42 s against
# a 5 s tolerance. More frames did hurt object recognition in our runs, so only
# the time route pays for it; every other format keeps the 384-frame setting.
TIME_FRAMES = 768

# --- Coarse-to-fine re-read for time questions ------------------------------
# The first pass usually lands in the right neighbourhood but not within 5 s of
# the event. So we use its answer to pick a window and look again: 192 frames
# over 240 s is one frame every 1.25 s, finer than the scoring tolerance.
#
# The window comes only from the model's own first answer, so nothing here
# depends on labels. A +/-120 s window worked better than +/-300 s on our
# calibration split: for a 5 s tolerance, resolution matters more than reach.
REFINE_FRAMES = 192
REFINE_HALF_S = 120.0

# Not every time question asks for a position on the timeline. Durations ("for
# how long", "adding together ...") and lists of time points are not a single
# moment, so re-reading around one instant makes no sense for them. They keep
# the first-pass answer.
_NOT_A_POSITION = re.compile(
    r"how much time|for how long|how long (?:is|was|does|did)|adding together"
    r"|total(?:ling)? |combined|time points|all the times|list all", re.I)

_HMS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")


def _first_hms(text) -> int | None:
    m = _HMS.search(str(text or ""))
    if not m:
        return None
    h, mm, ss = (int(x) for x in m.groups())
    return h * 3600 + mm * 60 + ss


def _refine(pipe, req, clip, coarse: str) -> tuple[str, bool]:
    """Second pass around the coarse estimate. Returns (answer, fired).

    Any failure keeps the coarse answer. The re-read is not free, though: it can
    also turn a correct coarse answer into a wrong one, which is why the window
    check below exists.
    """
    if _NOT_A_POSITION.search(str(req.question)):
        return coarse, False
    t_abs = _first_hms(coarse)
    if t_abs is None:
        return coarse, False
    start = float(getattr(req, "start_time", 0.0) or 0.0)
    t_local = t_abs - start                      # back to the clip's own clock
    try:
        vr = decord.VideoReader(str(clip), ctx=decord.cpu(0), num_threads=4)
        fps, nfull = float(vr.get_avg_fps()), len(vr)
        lo = max(0, int(round((t_local - REFINE_HALF_S) * fps)))
        hi = min(nfull - 1, int(round((t_local + REFINE_HALF_S) * fps)))
        if hi - lo < 2:
            return coarse, False
        step = max(1, (hi - lo) // REFINE_FRAMES)
        idx = list(range(lo, hi + 1, step))[:REFINE_FRAMES]
        batch = vr.get_batch(idx).asnumpy()
        del vr
        # Frame indices are absolute within the clip, so the timestamps Qwen sees
        # are already correct. We pass the original start_time along so that
        # the shift to the procedure timeline happens exactly once.
        shim = SimpleNamespace(qID=req.qID, question=req.question, start_time=start)
        out = pipe.answer(shim, clip, max_frames=REFINE_FRAMES,
                          clip_frames=nfull, frames=list(batch), frame_indices=idx)
    except Exception:
        log.exception("refine failed for %s; keeping the coarse answer", req.qID)
        return coarse, False
    t_out = _first_hms(out)
    if t_out is None:
        return coarse, False
    # If the refined answer falls outside the frames we just showed, the model
    # is not reading the window, it is guessing. On calibration none of these
    # answers were correct, so we keep the coarse one instead.
    span = (idx[-1] - idx[0]) / fps
    slack = max(5.0, span / max(len(idx) - 1, 1))      # one frame step, at least 5 s
    if not (idx[0] / fps - slack <= t_out - start <= idx[-1] / fps + slack):
        log.info("  %s refined answer left its window; keeping coarse", req.qID)
        return coarse, False
    return out, True


def log_environment(device: torch.device) -> None:
    log.info("--- Environment ---")
    log.info("  torch        : %s (CUDA %s)", torch.__version__, torch.version.cuda)
    arch = torch.cuda.get_arch_list() or (torch._C._cuda_getArchFlags() or "").split()
    log.info("  torch kernels: %s", " ".join(arch) or "unknown")
    if device.type != "cuda":
        log.warning("  No GPU visible - the platform always provides one.")
        return
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    log.info("  GPU          : %s (sm_%d%d)", torch.cuda.get_device_name(0), *cap)
    log.info("  VRAM         : %.1f / %.1f GiB free", free / 1024**3, total / 1024**3)


def run() -> int:
    t_start = time.monotonic()
    log.info("=== SurgLedger-FOCUS S1 — inference start ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_environment(device)

    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json contains no requests")
        return 1
    n = len(requests)
    # The container is not told which track it is running. SEGMENT allows 15 s
    # per question and PROCEDURE 30 s, and the clip length tells them apart:
    # SEGMENT windows are at most 300 s, PROCEDURE clips are close to full
    # procedures. Guessing too low only makes us shed frames earlier; guessing
    # too high can overrun, and a 20% overrun forfeits the batch.
    per_q = 15.0 if max((r.duration for r in requests), default=0.0) <= 600 else 30.0
    budget = budget_for(n, per_question_s=per_q)
    deadline = t_start + budget
    log.info("Batch of %d question(s) at %.0f s/question; self-imposed deadline %.0f s",
             n, per_q, budget)

    # /tmp is a mounted volume on the platform and we are not root, so it may
    # not be writable. Fall back to the home directory in that case.
    try:
        workdir = Path(tempfile.mkdtemp(prefix="focus-clips-"))
    except OSError:
        workdir = Path.home() / "focus-clips"
        workdir.mkdir(parents=True, exist_ok=True)
        log.warning("tempfile unusable; extracting clips to %s", workdir)
    clips = ClipSource(INPUT_PATH, workdir)

    # --- Load and warm up the model once, inside the setup allowance ---
    log.info("--- Loading model (once for the batch) ---")
    _ad = ADAPTER_DIR if (ADAPTER_DIR / "adapter_config.json").exists() else None
    log.info("adapter: %s", _ad or "none (base weights)")
    pipe = Pipeline(str(MODEL_DIR), device=device.type,
                    adapter_dir=str(_ad) if _ad else None)
    # set_fo_classes expects text. If the JSON file holds an object or a list
    # instead of a string we re-serialize it, because an exception here is
    # outside the per-question guard and would stop the whole batch. On any
    # error we keep the ten documented classes.
    try:
        raw_fo = (INPUT_PATH / "FO_definitions.json").read_text()
        try:
            parsed = json.loads(raw_fo)
        except ValueError:
            parsed = raw_fo
        pipe.set_fo_classes(parsed if isinstance(parsed, str) else json.dumps(parsed))
    except Exception:
        log.exception("FO_definitions unreadable; keeping the documented classes")
    pipe.warmup()
    log.info("Setup complete in %.2f s", time.monotonic() - t_start)

    # --- Inference ---
    responses, n_failed, n_degraded, n_time, n_refined = [], 0, 0, 0, 0
    for i, req in enumerate(requests, start=1):
        t0 = time.monotonic()
        # Routing only looks at the question text.
        try:
            fmt = route(req.question)
        except Exception:
            log.exception("router raised; falling back to the %d-frame path", MAX_FRAMES)
            fmt = None
        cap = TIME_FRAMES if fmt == "time" else MAX_FRAMES
        if fmt == "time":
            n_time += 1
        # Use fewer frames as the budget gets tight so that every question still
        # gets an answer. The logic is in decode.frames_for_budget, which has
        # its own tests.
        max_frames = frames_for_budget(deadline - t0, n - i + 1, cap,
                                       min_frames=MIN_FRAMES,
                                       degrade_at_s=DEGRADE_AT_S)
        if max_frames == 0:
            # No time left to look at the video: answer with the format prior.
            responses.append(Response(qID=req.qID, content=FALLBACK[route(req.question)],
                                      latency=0.0))
            log.warning("[%d/%d] %s: out of budget, emitting prior", i, n, req.qID)
            continue
        if max_frames < cap:
            n_degraded += 1

        try:
            clip = clips.get(req.qID)
            if clip is None:
                raise FileNotFoundError(f"no clip for {req.qID}")
            answer = pipe.answer(req, clip, max_frames=max_frames)
            # Only run the second pass while there is still enough budget left
            # for all remaining questions.
            if fmt == "time" and REFINE_FRAMES:
                spare = (deadline - time.monotonic()) / max(n - i, 1)
                if spare >= DEGRADE_AT_S:
                    answer, fired = _refine(pipe, req, clip, answer)
                    n_refined += fired
        except Exception:
            n_failed += 1
            answer = FALLBACK[route(req.question)]
            log.exception("[%d/%d] %s failed; emitting prior", i, n, req.qID)
        finally:
            clips.release(req.qID)

        latency = time.monotonic() - t0
        responses.append(Response(qID=req.qID, content=answer, latency=latency))
        log.info("[%d/%d] %s dur=%.0fs frames=%d %.2fs -> %r",
                 i, n, req.qID, req.duration, max_frames, latency, answer)

    total = time.monotonic() - t_start
    log.info("Answered %d (%d failed, %d degraded) in %.1f s (%.2f s/question)",
             len(responses), n_failed, n_degraded, total, total / max(n, 1))
    # Log clearly when the time route or the re-read never ran, since that
    # would otherwise go unnoticed in the output.
    if n_time:
        log.info("time route: %d of %d question(s) sampled at %d frames; "
                 "refined %d of them at %d frames over +/-%.0f s",
                 n_time, n, TIME_FRAMES, n_refined, REFINE_FRAMES, REFINE_HALF_S)
        if not n_refined:
            log.error("refinement fired on 0 of %d time question(s) -- the "
                      "coarse->fine path is INERT for this batch", n_time)
    else:
        log.error("time route fired on 0 of %d questions -- the dense-sampling "
                  "path is INERT for this batch", n)

    clips.close()
    shutil.rmtree(workdir, ignore_errors=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_items(responses, OUTPUT_PATH / "answer.json")
    log.info("Wrote %d response(s); total %.1f s of %.0f s allowed",
             len(responses), total, budget_for(n) / 0.75)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
