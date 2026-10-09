"""Fit the answer prior tables in surgledger/template_priors.py.

FOCUS questions come from a fixed set of templates, and for a few of them the
answer is almost always the same. The tables are not stored in this repository,
because they are derived from the challenge annotations. This script rebuilds
them from the PROCEDURE parquets and writes them back into the module, which
reproduces the tables used in the submission.

    python fit_template_priors.py --data /path/to/focus

Two tables are written:

  PRIORS         most common answer per (template, released answer_format) over
                 the train and test splits, for templates with at least 20 rows.
                 Only used for formats listed in pipeline.DEFER_FORMATS, which is
                 empty in the final submission.
  JUDGE_PRIORS   for the judge-graded formats (open_ended, multiple_choice),
                 templates whose most common answer covers at least 65% of at
                 least 8 training rows. Keyed by the format route() returns,
                 since that is all the pipeline knows at inference time. Answers
                 are compared with number words folded to digits and trailing
                 punctuation ignored, which is the leniency the judge allows.
"""
import argparse
import collections
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

SURGLEDGER = Path(__file__).resolve().parents[1] / "inference" / "resources" / "surgledger"
sys.path.insert(0, str(SURGLEDGER))
from template_priors import FO_NAMES, shell          # noqa: E402
from format_router import route                      # noqa: E402

MIN_N_PRIORS = 20
MIN_N_JUDGE = 8
MIN_RATE_JUDGE = 0.65
JUDGE_FORMATS = ("open_ended", "multiple_choice")
NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                "ten": "10"}


def read_rows(data: Path, split: str):
    out = []
    for ds in ("heico", "lapchole"):
        p = data / ds / "data" / "procedure" / f"{split}.parquet"
        t = pq.read_table(p)
        c = {k: t.column(k).to_pylist() for k in t.column_names}
        for i in range(t.num_rows):
            out.append((c["question"][i], str(c["answer"][i]), c["answer_format"][i]))
    return out


def norm(s: str) -> str:
    """Compare answers the way the judge does: case, spacing, trailing dot, digits."""
    s = re.sub(r"\s+", " ", str(s).strip().lower().rstrip("."))
    return NUMBER_WORDS.get(s, s)


def fit_priors(rows):
    fam = collections.defaultdict(list)
    for q, a, f in rows:
        fam[(shell(q), f)].append(a)
    out = {}
    for key, golds in fam.items():
        if len(golds) < MIN_N_PRIORS:
            continue
        ans, n = collections.Counter(golds).most_common(1)[0]
        out[key] = (ans, n / len(golds), len(golds))
    return out


def fit_judge_priors(train_rows):
    fam = collections.defaultdict(list)
    for q, a, f in train_rows:
        if f in JUDGE_FORMATS:
            fam[(shell(q), route(q))].append(a)
    out = {}
    for key, golds in fam.items():
        if len(golds) < MIN_N_JUDGE:
            continue
        normed, n = collections.Counter(norm(g) for g in golds).most_common(1)[0]
        rate = n / len(golds)
        if rate < MIN_RATE_JUDGE:
            continue
        # Write back one of the raw answers, not the normalised form, so the
        # emitted answer looks like the training data.
        raw = collections.Counter(g for g in golds if norm(g) == normed).most_common(1)[0][0]
        out[key] = (raw, rate, len(golds))
    return out


def render(name, table):
    lines = ["%s = {" % name]
    lines += ["    (%r, %r): (%r, %.3f, %d)," % (k[0], k[1], v[0], v[1], v[2])
              for k, v in sorted(table.items(), key=lambda kv: -kv[1][2])]
    lines.append("}")
    return "\n".join(lines)


def replace_block(text, name, block):
    """Swap the `NAME = {...}` assignment, which is a single line when empty."""
    start = text.index("%s = {" % name)
    end = text.index("}", start) + 1
    return text[:start] + block + text[end:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding heico/ and lapchole/")
    ap.add_argument("--out", default=str(SURGLEDGER / "template_priors.py"))
    a = ap.parse_args()
    data = Path(a.data)

    train = read_rows(data, "train")
    everything = train + read_rows(data, "test")
    priors = fit_priors(everything)
    judge = fit_judge_priors(train)
    print("PRIORS       %d templates over %d rows" % (len(priors), len(everything)))
    print("JUDGE_PRIORS %d templates over %d train rows" % (len(judge), len(train)))
    for (sh, fmt), (ans, rate, n) in sorted(judge.items(), key=lambda kv: -kv[1][2]):
        print("  n=%-4d rate=%.3f %-16s %s" % (n, rate, fmt, sh[:52]))

    out = Path(a.out)
    text = out.read_text(encoding="utf-8")
    text = replace_block(text, "PRIORS", render("PRIORS", priors))
    text = replace_block(text, "JUDGE_PRIORS", render("JUDGE_PRIORS", judge))
    out.write_text(text, encoding="utf-8")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
