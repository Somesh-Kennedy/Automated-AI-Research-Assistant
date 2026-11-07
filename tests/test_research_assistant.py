"""Offline tests: fake arXiv responses and synthetic abstracts, so no network or API key is needed."""
import pytest

from research_assistant import arxiv_client
from research_assistant.arxiv_client import Paper, build_query, fetch_papers, parse_feed
from research_assistant.clustering import cluster_papers, render_overview

VISION = ["convolutional network image classification pixels", "object detection bounding boxes images camera",
          "image segmentation convolutional pixels medical scans", "visual recognition images convolutional filters"]
SPEECH = ["speech recognition audio waveform acoustic", "acoustic model audio speaker phonemes",
          "speech synthesis audio waveform vocoder", "audio transcription acoustic speaker noise"]


FILLER = ["novel", "approach", "results", "benchmark", "method", "improves", "study", "framework", "dataset", "analysis"]


def make_papers(per_topic=15):
    """Two topics with overlapping, shuffled vocabulary so the groups are not trivially identical copies."""
    import random
    rng = random.Random(1)
    papers = []
    for tag, bank in (("v", VISION), ("s", SPEECH)):
        words = " ".join(bank).split()
        for i in range(per_topic):
            text = " ".join(rng.sample(words, 8) + rng.sample(FILLER, 4))
            papers.append(Paper(f"{tag}{i}", f"{tag.upper()} paper {i}", text, published="2025-01-01", link=f"http://x/{tag}{i}"))
    return papers


def entry(i):
    return (f"<entry><id>http://arxiv.org/abs/2501.{i:05d}v1</id><title>Paper {i}</title><summary> About {i}. </summary>"
            f"<published>2025-01-01T00:00:00Z</published><author><name>A {i}</name></author>"
            f"<link href='http://arxiv.org/abs/2501.{i:05d}v1'/></entry>")


def feed(ids):
    return f"<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'>{''.join(entry(i) for i in ids)}</feed>"


class FakeSession:
    """Serves `total` papers, honouring start / max_results like the real API."""
    def __init__(self, total):
        self.total, self.calls = total, []

    def get(self, url, params, timeout):
        self.calls.append(params)
        lo, hi = params["start"], min(params["start"] + params["max_results"], self.total)
        resp = type("R", (), {"text": feed(range(lo, hi)), "raise_for_status": lambda self: None})()
        return resp


# ------------------------------------------------------------------ retrieval
def test_build_query_uses_the_exact_phrase_or_every_word():
    assert build_query("graph neural networks") == 'all:"graph neural networks"'
    assert build_query("graph neural networks", exact=False) == "all:graph AND all:neural AND all:networks"
    with pytest.raises(ValueError):
        build_query("  ")


def test_parse_feed_extracts_fields():
    p = parse_feed(feed([7]))[0]
    assert p.arxiv_id == "2501.00007v1" and p.title == "Paper 7" and p.abstract == "About 7." and p.authors == ["A 7"]


def test_fetch_collects_100_papers_in_pages_and_waits_between_calls(monkeypatch):
    sleeps = []
    monkeypatch.setattr(arxiv_client.time, "sleep", sleeps.append)
    session = FakeSession(total=500)
    papers = fetch_papers("anything", 100, page_size=50, delay=3.0, session=session)
    assert len(papers) == 100 and len({p.arxiv_id for p in papers}) == 100
    assert [c["start"] for c in session.calls] == [0, 50] and sleeps == [3.0]


def test_fetch_stops_when_arxiv_has_fewer_papers(monkeypatch):
    monkeypatch.setattr(arxiv_client.time, "sleep", lambda s: None)
    assert len(fetch_papers("rare topic", 100, session=FakeSession(total=30))) == 30


def test_fetch_removes_duplicates_across_pages(monkeypatch):
    monkeypatch.setattr(arxiv_client.time, "sleep", lambda s: None)

    class Overlap(FakeSession):
        def get(self, url, params, timeout):                       # page 2 repeats the last paper of page 1
            start = max(params["start"] - (1 if params["start"] else 0), 0)
            return super().get(url, {**params, "start": start}, timeout)
    papers = fetch_papers("x", 60, page_size=30, session=Overlap(total=500))
    assert len(papers) == len({p.arxiv_id for p in papers})


# ------------------------------------------------------------------ clustering
def test_two_topics_are_separated_into_two_pure_clusters():
    result = cluster_papers(make_papers(), k=2)
    assert result.k == 2 and result.silhouette > 0.1
    for c in result.clusters:
        kinds = {p.arxiv_id[0] for p in c.papers}
        assert len(kinds) == 1, "a cluster mixed vision and speech papers"


def test_number_of_clusters_is_chosen_automatically():
    assert cluster_papers(make_papers()).k == 2


def test_cluster_terms_describe_the_theme():
    terms = {t for c in cluster_papers(make_papers(), k=2).clusters for t in c.top_terms}
    assert "audio" in terms and "images" in terms or "image" in terms


def test_every_paper_is_in_exactly_one_cluster_and_coords_match():
    papers = make_papers()
    result = cluster_papers(papers, k=2)
    assert sum(c.size for c in result.clusters) == len(papers) == len(result.labels)
    assert result.coords.shape == (len(papers), 2)


def test_too_few_papers_is_an_error():
    with pytest.raises(ValueError):
        cluster_papers(make_papers()[:3])


def test_overview_lists_themes_and_links():
    text = render_overview(cluster_papers(make_papers(), k=2))
    assert "30 papers grouped into 2 themes" in text and "### Theme 1" in text and "http://x/" in text


def test_rare_phrase_falls_back_to_all_words_query(monkeypatch):
    monkeypatch.setattr(arxiv_client.time, "sleep", lambda s: None)

    class Rare(FakeSession):
        def get(self, url, params, timeout):
            self.total = 5 if '"' in params["search_query"] else 500
            return super().get(url, params, timeout)
    session = Rare(total=5)
    assert len(fetch_papers("very rare topic", 100, session=session)) == 100
    assert any('"' in c["search_query"] for c in session.calls) and any(" AND " in c["search_query"] for c in session.calls)
