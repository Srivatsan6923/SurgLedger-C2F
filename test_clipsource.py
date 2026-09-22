"""Check that ClipSource finds clips in both input layouts.

On the platform the clips come as /input/batch-videos.zip, while the template's
local test data uses /input/plain/*.mp4. If only one layout worked, every clip
would be missing in the other case and the batch would silently fall back to
priors, since each miss is caught per question.

    python test_clipsource.py
"""

import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import inference  # noqa: E402


def _zip_layout(root: Path, members: list[str], tag: str) -> Path:
    inp = root / f"zipcase-{tag}"
    inp.mkdir()
    z = inp / "batch-videos.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for m in members:
            zf.writestr(m, b"\x00\x01fake-mp4-bytes")
    return inp


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # --- layout A: plain/ directory (template's local test data) ---
        inp = root / "dircase"
        (inp / "plain").mkdir(parents=True)
        (inp / "plain" / "q1.mp4").write_bytes(b"x")
        work = root / "wA"; work.mkdir()
        src = inference.ClipSource(inp, work)
        assert src.dir is not None, "should use the directory when present"
        assert src.get("q1") == inp / "plain" / "q1.mp4"
        assert src.get("missing") is None

        # --- layout B: batch-videos.zip, nested plain/ + overlayed/ ---
        inp = _zip_layout(root, ["batch-videos/plain/q1.mp4",
                                 "batch-videos/overlayed/q1.mp4",
                                 "batch-videos/plain/q2.mp4"], "B")
        work = root / "wB"; work.mkdir()
        src = inference.ClipSource(inp, work)
        assert src.dir is None and src.zf is not None
        assert len(src.index) == 2, src.index
        # plain must win over overlayed for the same qID
        assert "plain" in src.index["q1"], src.index["q1"]
        p = src.get("q1")
        assert p is not None and p.exists() and p.stat().st_size > 0
        src.release("q1")
        assert not p.exists(), "extracted clip must be deleted; a batch is tens of GB"
        assert src.get("nope") is None
        src.close()

        # --- layout C: flat zip, no subdirectories ---
        inp = _zip_layout(root, ["q7.mp4"], "C")
        work = root / "wC"; work.mkdir()
        src = inference.ClipSource(inp, work)
        assert src.get("q7") is not None
        src.close()

        # --- layout D: nothing at all -> None, never an exception ---
        inp = root / "empty"; inp.mkdir()
        work = root / "wD"; work.mkdir()
        src = inference.ClipSource(inp, work)
        assert src.get("q1") is None

    print("ClipSource self-check OK")


if __name__ == "__main__":
    main()
