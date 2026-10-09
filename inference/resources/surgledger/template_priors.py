"""Answer priors per question template.

FOCUS questions are generated from a fixed set of templates, and the template
can be recovered from the question text alone (replace times, numbers and FO
class names with placeholders, see `shell()`). For a few templates the correct
answer is almost always the same, and there a constant beats the model.

Two tables, each entry of the form

    (shell(question), format): (answer, rate, n_rows)

  * PRIORS: most common answer per template, used only for the formats listed
    in pipeline.DEFER_FORMATS.
  * JUDGE_PRIORS: templates in the judge-graded formats whose answer is nearly
    constant, keyed by the format route() returns.

The values are fitted on the challenge annotations and are not stored here;
`training/fit_template_priors.py` rebuilds them. While the tables are empty,
every question is answered by the model.
"""

import re

FO_NAMES = ('Sponge', 'Clip', 'Specimen bag', 'Specimen Bag', 'Silicone loop', 'Silicone Loop', 'External drain', 'External Drain', 'Needle', 'Gallstone', 'Specimen', 'Mesh')


def shell(question: str) -> str:
    """Reduce a question to its template key."""
    q = re.sub(r"\d{1,2}:[0-5]\d:[0-5]\d", "<T>", question)
    q = re.sub(r"\d+", "<N>", q)
    for c in FO_NAMES:
        q = q.replace(c, "<FO>").replace(c.lower(), "<FO>")
    return q[:88]


# (template, answer_format) -> (answer, rate, n)
PRIORS = {}

# (template, routed_format) -> (answer, rate, n), judge-graded formats only
JUDGE_PRIORS = {}


def judge_prior_for(question: str, fmt: str):
    """Prior for a judge-graded template, or None. See JUDGE_PRIORS."""
    return JUDGE_PRIORS.get((shell(question), fmt))


def prior_for(question: str, fmt: str):
    """Return (answer, rate, n) for this question's template, or None."""
    return PRIORS.get((shell(question), fmt))


def demo() -> None:
    q = "Does the Clip at 01:02:03 also appear at 02:03:04? Please answer with yes or no."
    assert shell(q) == "Does the <FO> at <T> also appear at <T>? Please answer with yes or no."
    # "Specimen bag" is listed before "Specimen", so it becomes one placeholder.
    assert shell("Is the Specimen bag visible?") == "Is the <FO> visible?"
    assert prior_for("not a real question at all", "binary") is None
    assert judge_prior_for("not a real question at all", "open_ended") is None
    print(f"template_priors self-check OK ({len(PRIORS)} + {len(JUDGE_PRIORS)} templates)")


if __name__ == "__main__":
    demo()
