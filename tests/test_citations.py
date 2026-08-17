from app.rag.citations import (
    ABSTAIN_SENTENCE,
    build_user_prompt,
    validate_answer,
)


def test_valid_citations_are_grounded():
    res = validate_answer("The policy allows 20 days [1] carried over [2].", num_sources=3)
    assert res.grounded is True
    assert res.abstained is False
    assert res.cited_indices == [1, 2]


def test_out_of_range_citations_are_stripped():
    res = validate_answer("Answer with a bad ref [9] and a good one [2].", num_sources=3)
    assert "[9]" not in res.answer
    assert res.cited_indices == [2]
    assert res.grounded is True


def test_uncited_answer_is_not_grounded():
    res = validate_answer("This is a confident answer with no citations.", num_sources=3)
    assert res.grounded is False
    assert res.abstained is False


def test_abstention_detected():
    res = validate_answer(ABSTAIN_SENTENCE, num_sources=3)
    assert res.abstained is True
    assert res.grounded is False


def test_prompt_numbers_sources_from_one():
    prompt = build_user_prompt("What is the PTO policy?", ["first passage", "second"])
    assert "[1] first passage" in prompt
    assert "[2] second" in prompt
    assert "What is the PTO policy?" in prompt
