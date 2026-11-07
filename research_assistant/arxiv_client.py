"""Fetch many papers for a topic from the public arXiv API (paginated, polite, de-duplicated)."""
import time
from dataclasses import dataclass, field

import feedparser
import requests

API = "https://export.arxiv.org/api/query"


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    published: str = ""
    link: str = ""

    @property
    def text(self) -> str:
        return f"{self.title}. {self.abstract}"


def build_query(topic: str, exact: bool = True) -> str:
    """exact: all:"graph neural networks" (the phrase).  not exact: all:graph AND all:neural AND all:networks."""
    words = [w for w in topic.replace('"', " ").split() if w]
    if not words:
        raise ValueError("topic is empty")
    return f'all:"{" ".join(words)}"' if exact else " AND ".join(f"all:{w}" for w in words)


def parse_feed(xml_text: str) -> list[Paper]:
    papers = []
    for e in feedparser.parse(xml_text).entries:
        papers.append(Paper(
            arxiv_id=e.id.rsplit("/abs/", 1)[-1],
            title=" ".join(e.title.split()),
            abstract=" ".join(e.summary.split()),
            authors=[a.name for a in getattr(e, "authors", [])],
            published=getattr(e, "published", ""),
            link=e.link,
        ))
    return papers


def fetch_papers(topic: str, max_results: int = 100, page_size: int = 50, delay: float = 3.0,
                 session=None) -> list[Paper]:
    """Return up to `max_results` unique papers, most relevant first. arXiv asks clients to wait ~3 s between calls."""
    session = session or requests.Session()
    papers = _fetch(build_query(topic), max_results, page_size, delay, session)
    if len(papers) < min(max_results, 30):              # the exact phrase is too rare: require all the words instead
        papers = _fetch(build_query(topic, exact=False), max_results, page_size, delay, session)
    return papers


def _fetch(query, max_results, page_size, delay, session) -> list[Paper]:
    seen, papers, start = set(), [], 0
    while len(papers) < max_results:
        size = min(page_size, max_results - len(papers))
        resp = session.get(API, params={"search_query": query, "start": start, "max_results": size,
                                        "sortBy": "relevance", "sortOrder": "descending"}, timeout=30)
        resp.raise_for_status()
        page = parse_feed(resp.text)
        if not page:
            break                                        # fewer papers exist than were requested
        for p in page:
            if p.arxiv_id not in seen:
                seen.add(p.arxiv_id)
                papers.append(p)
        start += len(page)
        if len(page) < size or len(papers) >= max_results:
            break
        time.sleep(delay)
    return papers[:max_results]
