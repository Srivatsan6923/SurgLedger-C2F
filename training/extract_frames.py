"""Build the JPEG frame cache that train_lora.py reads from.

One frame every 2 s (0.5 Hz) per video. Files are named by their index in 5 fps
clip units (index = round(t * 5)), which is the unit the samplers in decode.py
use, so a cached file name refers to the same moment as an inference frame
index. The frame rate is read from each video, since the source videos are not
5 fps.

We use ffmpeg rather than decord here: this is one sequential pass over many
hours of video, and ffmpeg's fps filter decodes once and only writes the frames
we keep.

    python extract_frames.py --roots <heico videos> <lapchole videos> --out frames/
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CLIP_FPS = 5.0          # evaluation clips are 5 fps; all indices are in these units
CACHE_HZ = 0.5          # one cached frame every 2 s -> 10 clip-index units apart
STRIDE = int(round(CLIP_FPS / CACHE_HZ))   # 10


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,nb_frames,width,height",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    num, den = (st.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    dur = float(j.get("format", {}).get("duration") or 0.0)
    return {"fps": fps, "duration": dur, "width": st.get("width"), "height": st.get("height")}


def extract_one(args):
    src, outdir, height, quality = args
    outdir = Path(outdir)
    marker = outdir / "_complete.json"
    if marker.exists():
        return {"video": Path(src).name, "status": "cached",
                "frames": json.loads(marker.read_text())["frames"]}
    t0 = time.time()
    try:
        info = probe(src)
    except Exception as e:
        return {"video": Path(src).name, "status": "probe_failed", "error": str(e)[:200]}
    if not info["fps"] or not info["duration"]:
        return {"video": Path(src).name, "status": "bad_metadata", "info": info}

    tmp = outdir.with_name(outdir.name + ".part")
    subprocess.run(["rm", "-rf", str(tmp)], check=False)
    tmp.mkdir(parents=True, exist_ok=True)
    # -vsync 0 keeps the output 1:1 with what the fps filter emits, so the N-th
    # file is exactly t = N / CACHE_HZ seconds. Without it ffmpeg may duplicate or
    # drop frames and the index-to-time mapping drifts.
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vf", f"fps={CACHE_HZ},scale=-2:{height}", "-vsync", "0",
           "-q:v", str(quality), str(tmp / "%08d.jpg")]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return {"video": Path(src).name, "status": "ffmpeg_failed",
                "error": (e.stderr or b"")[-300:].decode("utf-8", "replace")}

    # Rename ffmpeg's sequential output to clip-index names. File k (0-based) is
    # t = k / CACHE_HZ seconds -> clip index round(t * CLIP_FPS) = k * STRIDE.
    files = sorted(tmp.glob("*.jpg"))
    outdir.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(files):
        f.rename(outdir / f"{k * STRIDE:08d}.jpg")
    subprocess.run(["rm", "-rf", str(tmp)], check=False)

    meta = {"video": Path(src).name, "src_fps": info["fps"], "duration": info["duration"],
            "src_wh": [info["width"], info["height"]], "cache_hz": CACHE_HZ,
            "clip_fps": CLIP_FPS, "stride": STRIDE, "frames": len(files),
            "height": height, "seconds": time.time() - t0}
    marker.write_text(json.dumps(meta, indent=2))
    return {**meta, "status": "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="one or more directories holding videos (searched recursively)")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="restrict to these basenames (default: every video found)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--quality", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shortest-first", action="store_true", default=True)
    a = ap.parse_args()

    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".mpg", ".mpeg"}
    found = {}
    for root in a.roots:
        for p in Path(root).rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                found.setdefault(p.name, p)      # the basename is the parquet's join key
    want = set(a.videos) if a.videos else set(found)
    missing = sorted(want - set(found))
    todo = [(str(found[v]), str(Path(a.out) / Path(v).stem), a.height, a.quality)
            for v in sorted(want & set(found))]
    if a.shortest_first:
        todo.sort(key=lambda t: os.path.getsize(t[0]))   # small files finish first
    print(f"roots={a.roots}  found={len(found)}  requested={len(want)}  "
          f"to_extract={len(todo)}  MISSING={len(missing)}", flush=True)
    for m in missing[:10]:
        print(f"  !! missing {m}", flush=True)
    if missing:
        print(f"  ({len(missing)} missing total)", flush=True)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    ok = bad = 0; frames = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(extract_one, t): t[0] for t in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r["status"] in ("ok", "cached"):
                ok += 1; frames += r.get("frames", 0)
            else:
                bad += 1
                print(f"  FAIL {r['video']}: {r['status']} {str(r.get('error'))[:160]}", flush=True)
            if i % 5 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  [{i}/{len(todo)}] ok={ok} fail={bad} frames={frames} "
                      f"{el/60:.1f}min eta={(el/i)*(len(todo)-i)/60:.1f}min", flush=True)
    print(f"DONE ok={ok} fail={bad} missing={len(missing)} frames={frames} "
          f"minutes={(time.time()-t0)/60:.1f}", flush=True)
    # Exit non-zero if anything failed or is missing, so an incomplete cache
    # does not go unnoticed.
    sys.exit(1 if (bad or missing) else 0)


if __name__ == "__main__":
    main()
