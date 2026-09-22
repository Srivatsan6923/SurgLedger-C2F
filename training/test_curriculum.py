"""GPU-free checks for the data loading in train_lora.py.

Builds a fake frame cache (one empty folder per video, plus a stubbed image
reader), so load_rows / ProcedureSFT can be tested without the real cache or a
GPU. What is tested is the index arithmetic, which is where the mistakes that
do not raise errors tend to be.

Needs the challenge parquets, laid out as on Hugging Face:

    <data>/heico/data/{frame,segment,procedure}/train.parquet
    <data>/lapchole/data/{frame,segment,procedure}/train.parquet

    python test_curriculum.py --data /path/to/focus
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_lora as T  # noqa: E402

PARQUETS: list = []
CACHE = Path()


def build_fake_cache():
    """One empty folder per training video; load_rows only checks that it exists."""
    import pyarrow.parquet as pq
    stems = set()
    dur = {}
    for p in PARQUETS:
        t = pq.read_table(p, columns=["video", "timestamp_end"])
        for v, e in zip(t.column("video").to_pylist(), t.column("timestamp_end").to_pylist()):
            s = Path(v).stem
            stems.add(s)
            dur[s] = max(dur.get(s, 0.0), T.hms(e))
    CACHE.mkdir(parents=True, exist_ok=True)
    for s in stems:
        (CACHE / s).mkdir(exist_ok=True)
    return stems, dur


def install_stubs(dur):
    """Frame availability is computed instead of read from disk; images are stubs."""
    def _frames_for(self, video):
        stem = Path(video).stem
        if stem not in self._avail:
            n = int(round(dur[stem] * T.CLIP_FPS))
            grid = list(range(0, n + 1, T.CACHE_STRIDE))
            self._avail[stem] = (CACHE / stem, grid)
        return self._avail[stem]
    T.ProcedureSFT._frames_for = _frames_for

    class _Img:
        @staticmethod
        def open(path):
            class _I:
                def convert(self, _):
                    return np.zeros((36, 64, 3), dtype=np.uint8)
            return _I()
    import PIL
    PIL.Image = _Img
    sys.modules["PIL"].Image = _Img


def main():
    global PARQUETS, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding heico/ and lapchole/")
    a = ap.parse_args()
    data = Path(a.data)
    PARQUETS = [str(data / c / "data" / t / "train.parquet")
                for c in ("heico", "lapchole") for t in ("frame", "segment", "procedure")]
    CACHE = Path(tempfile.mkdtemp(prefix="fakecache-"))

    stems, dur = build_fake_cache()
    print(f"synthetic cache: {len(stems)} video dirs, "
          f"durations {min(dur.values()):.0f}-{max(dur.values()):.0f} s\n")
    install_stubs(dur)

    tr, va, counts, dropped = T.load_rows(
        PARQUETS, str(CACHE), CACHE / "val_videos.json",
        val_frac=0.15, seed=0, oversample_cap=6)
    print(f"train rows (after oversample) = {len(tr)}   val = {len(va)}   dropped = {dropped}")
    print(f"bucket counts before oversample: {counts}")

    # No video may be in both train and validation.
    val_v = set(json.loads((CACHE / "val_videos.json").read_text()))
    tr_v = {r["video"] for r in tr}
    assert not (tr_v & val_v), "TRAIN/VAL VIDEO LEAKAGE"
    print(f"val videos = {len(val_v)}, train videos = {len(tr_v)}, overlap 0")

    # Track mix that actually reaches the trainer.
    mix = {}
    for r in tr:
        mix[r["track"]] = mix.get(r["track"], 0) + 1
    print(f"track mix in train: {mix}\n")

    T.audit_alignment(tr, str(CACHE), 384, print)

    # PROCEDURE rows start at 00:00:00, so the rebase must not change them.
    import pyarrow.parquet as pq
    # Question templates repeat across videos, so compare whole rows.
    raw = set()
    for p in PARQUETS:
        if "procedure" not in p:
            continue
        t = pq.read_table(p, columns=["video", "question", "answer"])
        for v, q, ans in zip(t.column("video").to_pylist(),
                             t.column("question").to_pylist(),
                             t.column("answer").to_pylist()):
            raw.add((v, q, str(ans)))
    n_proc = n_same = 0
    for r in tr:
        if r["track"] != "procedure":
            continue
        n_proc += 1
        n_same += (r["video"], r["question"], str(r["answer"])) in raw
    print(f"PROCEDURE rows unchanged by rebase: {n_same}/{n_proc}")
    assert n_same == n_proc, "rebase altered PROCEDURE rows"

    # SEGMENT rows, on the other hand, must have moved, otherwise the rebase
    # silently did nothing.
    seg_raw = set()
    for p in PARQUETS:
        if "segment" not in p:
            continue
        t = pq.read_table(p, columns=["video", "question", "answer"])
        for v, q, ans in zip(t.column("video").to_pylist(),
                             t.column("question").to_pylist(),
                             t.column("answer").to_pylist()):
            seg_raw.add((v, q, str(ans)))
    # A row only changes if it contains a timestamp, and many SEGMENT questions
    # do not ("Which foreign object is inserted?"). So the check is: every row
    # with a timestamp has moved.
    n_seg = n_ts = n_moved = 0
    for r in tr:
        if r["track"] != "segment":
            continue
        n_seg += 1
        # Duration answers are elapsed times and must not shift, and their
        # questions usually carry no timestamp, so they are excluded here.
        is_dur = bool(T._DUR_Q.search(r["question"]))
        has_ts = (not is_dur and r["off"] > 0
                  and bool(T._TS_ANY.search(r["question"])
                           or T._TS_ANY.search(str(r["answer"]))))
        moved = (r["video"], r["question"], str(r["answer"])) not in seg_raw
        if has_ts:
            n_ts += 1
            n_moved += moved
        elif is_dur:
            g = T._TS_ANY.findall(str(r["answer"]))
            if len(g) == 1:
                assert T.hms(g[0]) <= r["win"] / T.CLIP_FPS + 1, (
                    "a DURATION answer exceeds its own window -- it was probably "
                    "rebased, or misclassified as a position")
    print(f"SEGMENT rows carrying a timestamp: {n_ts}/{n_seg}, all rebased: {n_moved}")
    assert n_moved == n_ts, f"{n_ts - n_moved} timestamped SEGMENT rows were not rebased"

    # Every rebased SEGMENT time answer has to fall inside its own window.
    outside = 0
    for r in tr:
        if r["track"] != "segment" or r["answer_format"] != "time":
            continue
        if T._DUR_Q.search(r["question"]):
            continue
        g = T._TS_ANY.findall(str(r["answer"]))
        if len(g) != 1:
            continue
        outside += not (0 <= T.hms(g[0]) * T.CLIP_FPS <= r["win"])
    print(f"rebased SEGMENT point-time answers outside their window: {outside}")
    assert outside == 0
    print("\nOK")


if __name__ == "__main__":
    main()
