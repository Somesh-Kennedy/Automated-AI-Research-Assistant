"""Group papers into themes: TF-IDF on title + abstract, LSA (SVD) to denoise, then K-Means.

The number of clusters is chosen by silhouette. On one narrow topic the themes overlap, so silhouette is low and tends to
keep rising with k; pass `k` explicitly when you want a fixed number of themes.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

from research_assistant.arxiv_client import Paper


@dataclass
class Cluster:
    id: int
    top_terms: list[str]
    papers: list[Paper]                    # ordered: closest to the cluster centre first
    @property
    def size(self) -> int:
        return len(self.papers)


@dataclass
class ClusterResult:
    k: int
    silhouette: float
    clusters: list[Cluster]
    coords: np.ndarray                     # 2-D positions for plotting
    labels: list[int]


def cluster_papers(papers: list[Paper], k: int | None = None, k_range: tuple[int, int] = (2, 8),
                   seed: int = 0, top_terms: int = 6) -> ClusterResult:
    if len(papers) < 4:
        raise ValueError("need at least 4 papers to cluster")
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
    try:
        X = vec.fit_transform([p.text for p in papers])
    except ValueError:                                  # min_df=2 left no terms (tiny / unrelated corpus)
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        X = vec.fit_transform([p.text for p in papers])
    terms = np.array(vec.get_feature_names_out())
    n_comp = min(10, X.shape[0] - 1, X.shape[1] - 1)
    Z = Normalizer().fit_transform(TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(X)) if n_comp >= 2 else X

    def fit(n: int) -> KMeans:
        return KMeans(n_clusters=n, n_init=10, random_state=seed).fit(Z)

    if k is None:
        lo, hi = k_range[0], min(k_range[1], len(papers) - 1)
        best = max((fit(n) for n in range(lo, hi + 1)),
                   key=lambda m: silhouette_score(Z, m.labels_, metric="cosine"))
    else:
        best = fit(k)
    labels = best.labels_
    sil = float(silhouette_score(Z, labels, metric="cosine")) if len(set(labels)) > 1 else 0.0

    clusters = []
    for c in range(best.n_clusters):
        idx = np.where(labels == c)[0]
        centre = np.asarray(Z[idx].mean(axis=0)).ravel()
        sim = np.asarray(Z[idx].dot(centre)).ravel()                             # higher = closer to the cluster centre
        order = idx[np.argsort(-sim)]
        weight = np.asarray(X[idx].mean(axis=0)).ravel()                          # mean TF-IDF weight names the theme
        top = terms[np.argsort(-weight)[:top_terms]].tolist()
        clusters.append(Cluster(c, top, [papers[i] for i in order]))
    clusters.sort(key=lambda c: -c.size)

    dims = min(2, X.shape[1] - 1)
    coords = TruncatedSVD(n_components=max(dims, 1), random_state=seed).fit_transform(X) if dims >= 1 else np.zeros((len(papers), 1))
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros_like(coords)])
    return ClusterResult(best.n_clusters, sil, clusters, coords, labels.tolist())


def render_overview(result: ClusterResult, per_cluster: int = 3) -> str:
    """Markdown overview handed to the writer agent (and shown in the app)."""
    total = sum(c.size for c in result.clusters)
    lines = [f"{total} papers grouped into {result.k} themes (silhouette {result.silhouette:.2f}).", ""]
    for n, c in enumerate(result.clusters, 1):
        lines.append(f"### Theme {n}: {', '.join(c.top_terms[:4])}  ({c.size} papers)")
        for p in c.papers[:per_cluster]:
            lines.append(f"- {p.title} ({p.published[:4]}) {p.link}")
        lines.append("")
    return "\n".join(lines)
