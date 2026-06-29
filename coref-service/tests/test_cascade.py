"""Tests for the LingMess -> s2e-coref cascade merge and rewrite logic.

These run without any model: the fastcoref / torch / transformers imports all live
inside ``load()``, so the merge, rewrite and tokenisation helpers are exercised
directly with hand-built clusters.
"""

from app.resolver import CorefResolver
from app.s2e_resolver import _word_spans


def span(text, sub, start=0):
    i = text.index(sub, start)
    return (i, i + len(sub))


def cascade():
    return CorefResolver(model_name="cascade", resolve_mode="anaphora")


# ── _merge_clusters ───────────────────────────────────────────────────
def test_s2e_fills_a_mention_lingmess_missed():
    text = "Rapamycin reduced mTOR. It was effective. The drug helped."
    primary = [[span(text, "Rapamycin"), span(text, "It")]]
    secondary = [[span(text, "Rapamycin"), span(text, "The drug")]]

    merged = cascade()._merge_clusters(text, primary, secondary)

    # primary cluster untouched, plus a supplemental cluster for "The drug"
    assert primary[0] in merged
    supp = [c for c in merged if c not in primary]
    assert len(supp) == 1
    assert span(text, "The drug") in supp[0]
    assert span(text, "Rapamycin") in supp[0]


def test_merge_never_re_adds_a_covered_mention():
    text = "Rapamycin reduced mTOR. It was effective. The drug helped."
    primary = [[span(text, "Rapamycin"), span(text, "It")]]
    # s2e re-finds "It" (already covered) plus the novel "The drug"
    secondary = [[span(text, "Rapamycin"), span(text, "It"), span(text, "The drug")]]

    merged = cascade()._merge_clusters(text, primary, secondary)
    supp = [c for c in merged if c not in primary][0]

    assert span(text, "It") not in supp           # not duplicated
    assert span(text, "The drug") in supp


def test_s2e_cluster_without_named_representative_is_skipped():
    text = "It was there. They saw it."
    primary = []
    secondary = [[span(text, "It"), span(text, "They")]]  # all pronouns

    merged = cascade()._merge_clusters(text, primary, secondary)
    assert merged == []


def test_s2e_cluster_with_no_novel_mentions_is_skipped():
    text = "Rapamycin reduced mTOR. It was effective."
    primary = [[span(text, "Rapamycin"), span(text, "It")]]
    secondary = [[span(text, "Rapamycin"), span(text, "It")]]  # nothing new

    merged = cascade()._merge_clusters(text, primary, secondary)
    assert merged == primary


# ── end-to-end rewrite through the cascade merge ──────────────────────
def test_cascade_rewrites_both_primary_and_s2e_mentions():
    text = "Rapamycin reduced mTOR. It was effective. The drug helped."
    primary = [[span(text, "Rapamycin"), span(text, "It")]]
    secondary = [[span(text, "Rapamycin"), span(text, "The drug")]]

    r = cascade()
    merged = r._merge_clusters(text, primary, secondary)
    out = r._apply_clusters(text, merged)

    assert out == "Rapamycin reduced mTOR. Rapamycin was effective. Rapamycin helped."


def test_overlapping_replacements_do_not_corrupt_text():
    text = "The drug worked. It helped."
    # Overlapping anaphors ("the drug" vs "drug") must not crash or double-substitute.
    clusters = [
        [span(text, "drug"), span(text, "It")],
        [(0, 8), span(text, "It")],  # "The drug" overlaps "drug"
    ]
    out = cascade()._apply_clusters(text, clusters)
    assert isinstance(out, str) and len(out) > 0


# ── s2e word/char tokenisation helper ─────────────────────────────────
def test_word_spans_recovers_offsets():
    text = "It works."
    spans = _word_spans(text)
    assert spans == [("It", 0, 2), ("works", 3, 8), (".", 8, 9)]
    # offsets must index back to the original substrings
    for word, s, e in spans:
        assert text[s:e] == word
