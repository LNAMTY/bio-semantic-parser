"""Unit tests for the instrumented PipelineRunner.

These run fully offline: coreference is disabled so no network/model is needed,
and the chunker step degrades gracefully if tiktoken cannot be loaded.
"""

import hashlib

from webapp.app.pipeline_runner import PipelineRunner, _find_coref_rewrites

SAMPLE = (
    "Rapamycin was administered daily. It reduced mTOR signalling. "
    "Copyright 2024 Example Journal. "
    "Methods We treated cells with the compound. "
    "Results The drug improved autophagy."
)


def _steps_by_id(result):
    return {s["id"]: s for s in result["steps"]}


def test_run_returns_all_six_steps_and_chunks():
    runner = PipelineRunner()
    result = runner.run(text=SAMPLE, use_coref=False)

    assert result["ok"] is True
    ids = [s["id"] for s in result["steps"]]
    assert ids == [1, 2, 3, 4, 5, 6]
    assert isinstance(result["chunks"], list)
    assert len(result["chunks"]) >= 1


def test_clean_step_removes_boilerplate():
    runner = PipelineRunner()
    result = runner.run(text=SAMPLE, use_coref=False)
    clean_step = _steps_by_id(result)[2]

    assert clean_step["detail"]["chars_removed"] > 0
    assert "copyright" not in clean_step["content"].lower()


def test_coref_step_is_skipped_when_disabled():
    runner = PipelineRunner()
    result = runner.run(text=SAMPLE, use_coref=False)
    coref_step = _steps_by_id(result)[3]

    assert coref_step["status"] == "skipped"


def test_sections_are_detected():
    runner = PipelineRunner()
    result = runner.run(text=SAMPLE, use_coref=False)
    section_step = _steps_by_id(result)[4]

    names = [s["section"] for s in section_step["detail"]["sections"]]
    assert "methods" in names or "results" in names


def test_document_id_is_deterministic_sha256():
    runner = PipelineRunner()
    result = runner.run(text=SAMPLE, use_coref=False)

    expected = hashlib.sha256(SAMPLE.encode("utf-8")).hexdigest()
    assert result["document_id"] == expected


def test_chunks_carry_metadata():
    runner = PipelineRunner()
    result = runner.run(text=SAMPLE, source_name="unit-test", use_coref=False)

    for i, chunk in enumerate(result["chunks"]):
        assert chunk["document_id"] == result["document_id"]
        assert chunk["source_name"] == "unit-test"
        assert chunk["position"] == i
        assert "section" in chunk


def test_find_coref_rewrites_detects_changed_sentences():
    before = "Rapamycin was given. It reduced mTOR activity."
    after = "Rapamycin was given. Rapamycin reduced mTOR activity."

    rewrites = _find_coref_rewrites(before, after)

    assert len(rewrites) == 1
    assert rewrites[0]["before"] == "It reduced mTOR activity."
    assert rewrites[0]["after"] == "Rapamycin reduced mTOR activity."
