"""Work out the expected answer format from the question text, and serialize to it.

`Request` does not include an answer_format field, but every FOCUS question
states its format in the text ("Please answer with yes or no", "in hh:mm:ss",
...). So routing is a simple keyword match.

This matters because the official evaluator scores a question as wrong when
`fmt.read(response.content)` raises. A correct answer in the wrong shape still
counts as wrong, so everything we emit goes through `serialize()` first.

Self-check: python format_router.py
"""

import re

# Checked in order, first match wins. The order matters: a multiple-choice
# question also mentions "quadrant", and a percentage question also asks "how many".
_RULES = [
    ("binary", ("yes or no",)),
    ("time", ("hh:mm:ss",)),
    ("percentage", ("xx%", "in %,")),
    # "please select one answer" and "please select none, one or multiple answers"
    ("multiple_choice", ("please select",)),
    ("fo_class", ("class name", "class name(s)")),
    ("number", ("provide a number", "provide the number")),
]

# Fallback answer per format: the most common answer for that format in the
# training data. Used when the model fails, runs out of time or returns
# something unparseable. An empty answer is always wrong, the prior is
# sometimes right.
FALLBACK = {
    "binary": "no",
    "number": "1",
    "fo_class": "none",
    "time": "00:24:46",
    "percentage": "2.5",
    "multiple_choice": "bottom/left",
    "open_ended": "none",
}

_TIME_RE = re.compile(r"\b(\d{1,2}):([0-5]\d):([0-5]\d)\b")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_QUADRANT_RE = re.compile(r"\b(top|bottom)\s*[/\-]?\s*(left|right)\b", re.I)
_MULTI_TIME_RE = re.compile(r"\ball\b|\beach\b|\btime ?points\b|\btimestamps\b|\blist\b", re.I)


def wants_multiple_times(question: str) -> bool:
    """Does this time question expect more than one timestamp?

    `Time.compare` in the focus package rejects a length mismatch before it
    compares any value, so an extra timestamp is always scored wrong. Almost all
    time answers are a single timestamp, and the multi-timestamp questions all
    contain one of these cue words, so we default to one timestamp otherwise.
    """
    return bool(_MULTI_TIME_RE.search(question or ""))

# The 10 documented classes. New ones may show up at test time through
# /input/FO_definitions.json, so callers should pass `valid` from that file
# instead of relying on this default.
DEFAULT_FO_CLASSES = (
    "Sponge", "Clip", "Specimen Bag", "Silicone Loop", "External Drain",
    "Needle", "Gallstone", "Specimen", "Mesh", "Absorbable Hemostatic Agent",
)


# Fallback rules for questions that do not state their format. All challenge
# questions do, but the sample questions in the submission template do not
# ("Is a foreign object visible in the scene?"), and without these rules they
# would all become open_ended. Only used when _RULES finds nothing.
_SHAPE_RULES = [
    ("number", (r"^\s*how many\b", r"\bhow many\b.*\?")),
    ("time", (r"^\s*(at what time|when)\b", r"\bat which time\b")),
    ("percentage", (r"\bwhat percentage\b", r"\bwhat % \b")),
    ("binary", (r"^\s*(is|are|was|were|does|do|did|has|have|can|could|should)\b",)),
    ("fo_class", (r"^\s*(which|what)\b.*\bforeign object", r"\bwhich (fo|object|class)")),
]


def route(question: str) -> str:
    """Return the answer_format string implied by the question text."""
    q = question.lower()
    for fmt, needles in _RULES:
        if any(n in q for n in needles):
            return fmt
    # The shape rules only apply to questions with no instruction at all. Real
    # open_ended questions also end with a "Please ..." clause ("Please provide
    # an anatomical location"), and applying shape rules to them sends some of
    # them to the wrong format.
    if "please" not in q:
        for fmt, patterns in _SHAPE_RULES:
            if any(re.search(p, q) for p in patterns):
                return fmt
    return "open_ended"


def _seconds_to_hms(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def serialize(raw: str, fmt: str, valid_fo: tuple[str, ...] = DEFAULT_FO_CLASSES,
              question: str = "") -> str:
    """Turn free-form model output into something `fmt.read()` accepts.

    If nothing usable is found we return the format's prior rather than junk.
    """
    text = (raw or "").strip()
    if not text:
        return FALLBACK[fmt]

    if fmt == "binary":
        low = text.lower()
        if re.search(r"\byes\b", low):
            return "yes"
        if re.search(r"\bno\b", low):
            return "no"
        return FALLBACK[fmt]

    if fmt == "time":
        # Keep the order in which timestamps appear rather than sorting. When
        # only one is allowed, the first one the model states is its answer;
        # smaller numbers later in the text are often side remarks.
        multi = wants_multiple_times(question)
        hits = _TIME_RE.findall(text)
        if hits:
            secs = list(dict.fromkeys(int(h) * 3600 + int(m) * 60 + int(s)
                                      for h, m, s in hits))
            secs = sorted(secs) if multi else secs[:1]
            return ", ".join(_seconds_to_hms(s) for s in secs)
        # Plain seconds. Qwen3-VL labels frames in seconds, and the base model
        # was bad at converting to hh:mm:ss itself, so if it answers in seconds
        # we do the conversion here.
        nums = [float(m.group()) for m in _NUM_RE.finditer(text)]
        nums = list(dict.fromkeys(n for n in nums if 0 <= n <= 24 * 3600))
        if nums:
            nums = sorted(nums) if multi else nums[:1]
            return ", ".join(_seconds_to_hms(s) for s in nums)
        return FALLBACK[fmt]

    if fmt == "number":
        m = _NUM_RE.search(text)
        if not m:
            # Spelled-out numbers ("Five.") fall back to the prior. We tried a
            # word parser, but the fine-tuned model always answers in digits,
            # so it never changed an answer and was removed.
            return FALLBACK[fmt]
        # Number.read() expects a non-negative integer.
        return str(max(0, int(round(float(m.group())))))

    if fmt == "percentage":
        m = _NUM_RE.search(text.replace("%", " "))
        return str(float(m.group())) if m else FALLBACK[fmt]

    if fmt == "multiple_choice":
        # Some questions allow "none, one or multiple" quadrants, so keep every
        # distinct hit, in top/bottom-left/right order.
        hits = _QUADRANT_RE.findall(text)
        if not hits:
            # multiple_choice is graded by the LLM judge (verify() only checks
            # the length), and the judge accepts paraphrases like "lower left".
            # So the model's own words are a better bet than a constant.
            return "none" if "none" in text.lower() else text[:300]
        order = ["top/left", "top/right", "bottom/left", "bottom/right"]
        found = {f"{v.lower()}/{h.lower()}" for v, h in hits}
        return ", ".join(q for q in order if q in found)

    if fmt == "fo_class":
        low = text.lower()
        # Match longest names first so "Specimen Bag" is not read as "Specimen".
        # Each match is blanked out (keeping offsets) so a shorter class can
        # still match somewhere else in the text.
        #
        # This matters because "Specimen" and "Specimen Bag" are separate
        # classes and some answers list both ("Specimen, Specimen bag, Sponge").
        # FOClass compares sets exactly, so dropping either one makes the
        # answer wrong.
        work = list(low)
        found = []
        for c in sorted(valid_fo, key=len, reverse=True):
            cl = c.lower()
            pos = "".join(work).find(cl)
            if pos >= 0:
                found.append((pos, c))
                work[pos:pos + len(cl)] = "\x00" * len(cl)   # blank it, keep offsets
        if not found:
            return "none"
        # Keep the order of appearance. FOClass.read() returns a set, so order
        # does not matter for real fo_class questions. But questions like "In
        # what chronological order do the classes first appear?" also route
        # here, and those are graded by the judge, which does read order.
        #
        # Sort by the matched position, not low.index(): for "Specimen" that
        # would find the offset inside "specimen bag" and reorder the answer.
        kept = [c for _, c in sorted(found, key=lambda t: t[0])]
        return ", ".join(kept)

    return text[:300]  # open_ended: OpenEnded enforces max_length=300


def demo() -> None:
    cases = [
        ("Please answer with yes or no.", "binary"),
        ("Please provide the answer in hh:mm:ss.", "time"),
        ("In %, how many of the frames of this video contain a Clip? Please provide the answer in the format xx%.", "percentage"),
        ("Please select one answer: top/left, top/right...", "multiple_choice"),
        ("Please provide a class name or answer with none.", "fo_class"),
        ("How many Clip(s) are inserted? Please provide a number.", "number"),
        ("In which abdominal quadrant is the external drain placed? Please provide an anatomical location.", "open_ended"),
    ]
    for q, expected in cases:
        assert route(q) == expected, f"route({q!r}) -> {route(q)!r}, want {expected!r}"

    # Questions with no stated format, like the template's sample questions.
    bare = [
        ("Is a foreign object visible in the scene?", "binary"),
        ("How many foreign objects are visible?", "number"),
        ("Is the foreign object in contact with tissue?", "binary"),
        ("At what time was the sponge removed?", "time"),
        ("Which foreign object was left behind?", "fo_class"),
        ("Describe the surgical scene.", "open_ended"),
    ]
    for q, expected in bare:
        assert route(q) == expected, f"bare route({q!r}) -> {route(q)!r}, want {expected!r}"

    assert serialize("I think the answer is Yes, clearly.", "binary") == "yes"
    # No digits -> prior (see the note in the number branch).
    assert serialize("There appear to be two needles", "number") == "1"
    assert serialize("There are 2 needles", "number") == "2"
    assert serialize("about 3.7 items", "number") == "4"
    # A single timestamp unless the question asks for several (then sorted).
    assert serialize("at 01:23:45 and again 00:10:00", "time") == "01:23:45"
    assert serialize("at 01:23:45 and again 00:10:00", "time",
                     question="List all time points") == "00:10:00, 01:23:45"
    assert serialize("3847 seconds, about 64 minutes in", "time") == "01:04:07"
    assert wants_multiple_times("At what time points does each Clip appear?")
    assert not wants_multiple_times("At what time is the Sponge inserted?")
    assert serialize("no timestamp here", "time") == FALLBACK["time"]
    # Plain seconds are converted here.
    assert serialize("3847", "time") == "01:04:07"
    assert serialize("The sponge appears at 3847.5 seconds", "time") == "01:04:08"
    assert serialize("120, 3600", "time",
                 question="all time points") == "00:02:00, 01:00:00"
    assert serialize("120, 3600", "time") == "00:02:00"
    # hh:mm:ss wins when present, and out-of-range numbers are ignored.
    assert serialize("00:10:00 (600 seconds)", "time") == "00:10:00"
    assert serialize("999999", "time") == FALLBACK["time"]
    assert serialize("I see a Sponge and a Clip", "fo_class") == "Sponge, Clip"
    # Order of appearance is kept.
    assert serialize("Sponge first, then Clip", "fo_class") == "Sponge, Clip"
    assert serialize("Clip first, then Sponge", "fo_class") == "Clip, Sponge"
    assert serialize("nothing visible", "fo_class") == "none"
    assert serialize("the Specimen Bag is there", "fo_class") == "Specimen Bag"
    # "Specimen" and "Specimen Bag" are different classes; both must survive.
    assert serialize("Specimen, Specimen bag", "fo_class") == "Specimen, Specimen Bag"
    assert serialize("Specimen Bag, Specimen, Clip", "fo_class") == "Specimen Bag, Specimen, Clip"
    # ...and "Specimen" alone must stay "Specimen".
    assert serialize("Specimen", "fo_class") == "Specimen"
    assert serialize("roughly 4.5%", "percentage") == "4.5"
    assert serialize("it is in the bottom right", "multiple_choice") == "bottom/right"
    assert serialize("", "binary") == "no"
    print("format_router self-check OK")


if __name__ == "__main__":
    demo()
