from app.rag.hybrid import label_retrieval, reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_lists():
    # "b" is ranked well by both lists; it should win.
    lexical = ["a", "b", "c"]
    semantic = ["b", "d", "a"]
    fused = reciprocal_rank_fusion([lexical, semantic], k=60)
    ids = [doc for doc, _ in fused]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_scores_are_descending():
    fused = reciprocal_rank_fusion([["x", "y", "z"]], k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_top_rank_beats_lower_rank():
    # Same single list: rank 0 must outscore rank 2.
    fused = dict(reciprocal_rank_fusion([["p", "q", "r"]], k=60))
    assert fused["p"] > fused["q"] > fused["r"]


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_label_retrieval():
    assert label_retrieval("d1", {"d1"}, {"d1"}) == "hybrid"
    assert label_retrieval("d2", set(), {"d2"}) == "vector"
    assert label_retrieval("d3", {"d3"}, set()) == "lexical"
