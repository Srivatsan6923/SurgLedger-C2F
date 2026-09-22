"""Frame sampling for FOCUS clips.

Evaluation clips are 5 fps with a keyframe every 5 s, so every 25th frame is a
keyframe. If we only sample multiples of 25, decord never has to reconstruct
inter-frames, which is what makes it affordable to cover a clip of several
hours inside the 30 s per-question budget.

Self-check: python decode.py
"""

import re

CLIP_FPS = 5.0
KEYFRAME_EVERY_S = 5.0
KEYFRAME_STRIDE = int(CLIP_FPS * KEYFRAME_EVERY_S)  # 25 frames


def plan_indices(n_frames: int, max_frames: int) -> list[int]:
    """Choose <= max_frames indices spanning the clip, snapped to keyframes.

    Long clips: evenly spaced keyframes (multiples of KEYFRAME_STRIDE).
    Very short clips (fewer than 64 keyframes): an even sweep over all frames,
    where decoding a few off-grid frames costs almost nothing.
    """
    if n_frames <= 0:
        return [0]
    if max_frames <= 0:
        return [0]

    keyframes = list(range(0, n_frames, KEYFRAME_STRIDE)) or [0]

    if len(keyframes) <= max_frames:
        # Even if the budget allows more frames than there are keyframes, we
        # stay on the grid. A keyframe decodes in about 4 ms, a frame in the
        # middle of a GOP in 13-16 ms, because the decoder has to rebuild it
        # from the previous I-frame. The grid is also fine for timing: keyframes
        # are 5 s apart and the time tolerance is 5 s. Finer resolution is the
        # job of the re-read in inference.py.
        if len(keyframes) >= 64:
            return keyframes
        if n_frames <= max_frames:
            return list(range(n_frames))
        step = (n_frames - 1) / (max_frames - 1) if max_frames > 1 else 0
        return sorted({int(round(i * step)) for i in range(max_frames)})

    # Subsample the keyframe grid evenly, keeping first and last.
    if max_frames == 1:
        return [keyframes[len(keyframes) // 2]]
    step = (len(keyframes) - 1) / (max_frames - 1)
    return sorted({keyframes[int(round(i * step))] for i in range(max_frames)})


_TS_RE = re.compile(r"\b(\d{1,2}):([0-5]\d):([0-5]\d)\b")


def parse_timestamps(question: str) -> list[float]:
    """Timestamps (in seconds, original timeline) mentioned in the question.

    Many binary questions refer to a specific moment ("Does the Silicone loop at
    04:44:08 also appear at 02:07:00?"), and uniform sampling will almost never
    land on it. Returned deduplicated, in order of first appearance.
    """
    seen: dict[float, None] = {}
    for h, m, s in _TS_RE.findall(question or ""):
        seen.setdefault(int(h) * 3600 + int(m) * 60 + int(s), None)
    return list(seen)


def plan_indices_targeted(n_frames: int, max_frames: int, timestamps_s: list[float],
                          start_time: float = 0.0, window_s: float = 45.0,
                          targeted_share: float = 0.6) -> list[int]:
    """Frame indices concentrated around the moments the question mentions.

    `targeted_share` of the budget goes to dense sampling within +/- `window_s`
    of each timestamp, and the rest stays on the uniform keyframe grid so the
    model still sees the whole clip. With no timestamps this is the same as
    `plan_indices`.
    """
    if n_frames <= 0 or max_frames <= 0:
        return [0]
    if not timestamps_s:
        return plan_indices(n_frames, max_frames)

    # Question timestamps are on the original timeline; the clip starts at start_time.
    centres = []
    for t in timestamps_s:
        f = int(round((t - start_time) * CLIP_FPS))
        if -n_frames < f < 2 * n_frames:      # ignore values far outside the clip
            centres.append(min(max(f, 0), n_frames - 1))
    if not centres:
        return plan_indices(n_frames, max_frames)

    n_targeted = max(len(centres), int(max_frames * targeted_share))
    n_targeted = min(n_targeted, max_frames)
    per = max(1, n_targeted // len(centres))
    half = int(window_s * CLIP_FPS)

    picked: set[int] = set()
    for c in centres:
        lo, hi = max(0, c - half), min(n_frames - 1, c + half)
        if hi <= lo:
            picked.add(lo)
            continue
        step = (hi - lo) / max(per - 1, 1) if per > 1 else 0
        for i in range(per):
            picked.add(int(round(lo + i * step)))

    # Fill the rest from the uniform grid for context outside those moments.
    for idx in plan_indices(n_frames, max(max_frames - len(picked), 1)):
        if len(picked) >= max_frames:
            break
        picked.add(idx)

    out = sorted(picked)
    if len(out) > max_frames:
        # Trim evenly instead of truncating, so both ends are kept.
        step = (len(out) - 1) / (max_frames - 1)
        out = sorted({out[int(round(i * step))] for i in range(max_frames)})
    return out


def frame_to_original_time(frame_index: int, start_time: float) -> float:
    """Clip frame index -> time on the original procedure timeline.

    The clip is already cut to [start_time, end_time], so its frame 0 is
    start_time. No need to read the burned-in clock.
    """
    return start_time + frame_index / CLIP_FPS


def budget_for(n_questions: int, per_question_s: float = 30.0,
               setup_s: float = 120.0, safety: float = 0.75) -> float:
    """Total wall-clock time we allow ourselves for the batch.

    The platform allows `setup_s + n * per_question_s` and drops questions once
    that is exceeded (a 20% overrun loses the whole batch). `safety` keeps us
    well below the limit.
    """
    return (setup_s + n_questions * per_question_s) * safety


def frames_for_budget(remaining_s: float, questions_left: int, max_frames: int,
                      min_frames: int = 8, degrade_at_s: float = 20.0,
                      give_up_at_s: float = 2.0) -> int:
    """How many frames the next question can afford. 0 means "use the prior".

    Kept out of the inference loop so it can be tested on its own; a mistake
    here can cost the whole batch.
    """
    if questions_left <= 0:
        return 0
    per_q = remaining_s / questions_left
    if per_q <= give_up_at_s:
        return 0
    if per_q >= degrade_at_s:
        return max_frames
    return max(min_frames, int(max_frames * per_q / degrade_at_s))


def demo() -> None:
    # Long clip: 16,749 s -> 83,746 frames.
    idx = plan_indices(83746, 64)
    assert len(idx) == 64, len(idx)
    assert all(i % KEYFRAME_STRIDE == 0 for i in idx), "long clips must hit keyframes"
    assert idx[0] == 0 and idx[-1] <= 83745

    # Longest possible clip.
    idx = plan_indices(88900, 32)
    assert len(idx) == 32 and all(i % KEYFRAME_STRIDE == 0 for i in idx)

    # Short clip (309 s = 1545 frames = 62 keyframes) with room for 64 -> denser.
    idx = plan_indices(1545, 64)
    assert len(idx) == 64, len(idx)

    # Very short clip: fewer frames than budget -> take everything.
    assert plan_indices(10, 64) == list(range(10))
    assert plan_indices(0, 64) == [0]
    assert len(plan_indices(83746, 1)) == 1

    # When the whole keyframe grid fits in the budget, we must still return
    # keyframes only, not sweep every frame.
    #   1,499 s = 7,495 frames = 300 keyframes, budget 384.
    lap = plan_indices(7495, 384)
    assert all(i % KEYFRAME_STRIDE == 0 for i in lap), "off-grid sweep is back"
    assert len(lap) == 300, len(lap)
    #   6,399 s = 31,995 frames = 1,280 keyframes -> subsample the grid.
    hei = plan_indices(31995, 384)
    assert all(i % KEYFRAME_STRIDE == 0 for i in hei), "off-grid subsample"
    #   Below 64 keyframes the dense sweep is still used.
    assert plan_indices(1545, 64) != list(range(0, 1545, KEYFRAME_STRIDE))

    # Timestamp arithmetic: 5 fps, offset by the clip's own start.
    assert frame_to_original_time(0, 132.5) == 132.5
    assert frame_to_original_time(25, 100.0) == 105.0  # one keyframe = 5 s
    assert frame_to_original_time(5, 0.0) == 1.0

    # Budget: 20 questions -> 120 + 600 = 720 s, times safety.
    assert budget_for(20) == 540.0

    # --- budget guard ---------------------------------------------------
    F = 128
    assert frames_for_budget(540, 20, F) == F           # fresh batch: full frames
    assert frames_for_budget(400, 20, F) == F           # 20 s/q exactly at threshold
    assert frames_for_budget(200, 20, F) == 64          # 10 s/q -> half
    assert frames_for_budget(100, 20, F) == 32          # 5 s/q -> quarter
    assert frames_for_budget(60, 20, F) == 19           # 3 s/q -> near the floor
    assert frames_for_budget(40, 20, F) == 0            # 2 s/q is exactly give-up
    assert frames_for_budget(20, 20, F) == 0            # 1 s/q -> emit prior
    assert frames_for_budget(-100, 5, F) == 0           # already over: never negative
    assert frames_for_budget(1000, 0, F) == 0           # no questions left
    assert frames_for_budget(1e6, 1, F) == F            # huge budget capped at max
    # More time never gives fewer frames.
    prev = -1
    for r in range(0, 600, 10):
        cur = frames_for_budget(r, 20, F)
        assert cur >= prev, f"non-monotonic at remaining={r}"
        prev = cur
    # Never above the cap, never below the floor unless giving up.
    for r in range(0, 2000, 7):
        for left in (1, 5, 20, 50):
            v = frames_for_budget(r, left, F)
            assert v == 0 or 8 <= v <= F, (r, left, v)

    # --- timestamp parsing ------------------------------------------------
    assert parse_timestamps("Does the loop at 04:44:08 also appear at 02:07:00?") == [17048.0, 7620.0]
    assert parse_timestamps("no times here") == []
    assert parse_timestamps("at 00:00:00") == [0.0]
    assert parse_timestamps("twice 01:02:03 and 01:02:03") == [3723.0]   # deduped
    assert parse_timestamps("bad 99:99:99") == []                        # not a valid time
    assert parse_timestamps(None) == []

    # --- targeted sampling ------------------------------------------------
    N = 29945           # a 5,989 s clip at 5 fps
    # No timestamps -> identical to the uniform path.
    assert plan_indices_targeted(N, 128, []) == plan_indices(N, 128)

    # One timestamp: most frames land near it, and we stay within budget.
    idx = plan_indices_targeted(N, 128, [3000.0])
    assert len(idx) <= 128
    c = int(3000.0 * CLIP_FPS)
    near = [i for i in idx if abs(i - c) <= 45 * CLIP_FPS]
    assert len(near) >= 64, f"expected dense coverage near the cited time, got {len(near)}"
    assert min(idx) >= 0 and max(idx) < N
    assert idx == sorted(set(idx)), "indices must be sorted and unique"

    # Two timestamps: both get covered.
    idx = plan_indices_targeted(N, 128, [1000.0, 5000.0])
    for t in (1000.0, 5000.0):
        c = int(t * CLIP_FPS)
        assert any(abs(i - c) <= 45 * CLIP_FPS for i in idx), f"missed {t}"

    # Edges and out-of-range values must clamp, never crash or go negative.
    for ts in ([0.0], [5988.0], [999999.0], [-50.0], [0.0, 5988.0]):
        v = plan_indices_targeted(N, 64, ts)
        assert v and len(v) <= 64 and min(v) >= 0 and max(v) < N, (ts, len(v))

    # More timestamps than budget: still bounded, still one frame each.
    many = [float(x) for x in range(0, 5900, 50)]      # 118 timestamps
    v = plan_indices_targeted(N, 32, many)
    assert len(v) <= 32 and min(v) >= 0 and max(v) < N

    # Short clip, and a non-zero start_time.
    assert plan_indices_targeted(50, 64, [5.0]) and max(plan_indices_targeted(50, 64, [5.0])) < 50
    off = plan_indices_targeted(N, 64, [3600.0], start_time=3000.0)
    c = int((3600.0 - 3000.0) * CLIP_FPS)
    assert any(abs(i - c) <= 45 * CLIP_FPS for i in off), "start_time offset ignored"

    print("decode self-check OK")


if __name__ == "__main__":
    demo()
