# Automated AI Research Assistant

Give it a research topic. It pulls **~100 papers from arXiv**, **clusters them into themes**, and two CrewAI agents turn
those themes into a structured, referenced summary.

```
topic ─► arXiv API (paginated, 50 per request, 3 s apart) ─► TF-IDF ─► LSA (10 dims) ─► K-Means (k by silhouette)
      ─► researcher agent reads the theme overview ─► writer agent drafts the markdown summary + references
```

## Contents
[How it works](#how-it-works) · [Example run](#example-run) · [Run it](#run-it) · [Configuration](#configuration) ·
[Tests](#tests) · [Layout](#layout) · [Limitations](#limitations)

## How it works

**1. Retrieval** (`research_assistant/arxiv_client.py`). The public arXiv API returns at most a page of results per call,
so the client asks for 50 at a time, waits 3 seconds between calls (arXiv's request), and stops at the requested total.
100 papers take two requests. The query is the **exact phrase** first (`all:"retrieval augmented generation"`), sorted by
relevance. If fewer than 30 papers match, it retries with "all the words must appear". Results are de-duplicated by arXiv
id across pages. Each paper keeps its id, title, abstract, authors, date and link.

**2. Vectorizing** (`research_assistant/clustering.py`). Each paper's title and abstract become a **TF-IDF** vector: words
that are frequent in one paper but rare overall get high weight. Single words and word pairs are used, English stop words
are dropped, and words that appear in more than 80% of papers are ignored (they cannot separate themes).

**3. Denoising.** The TF-IDF matrix is wide and sparse, and clustering it directly gave a silhouette of 0.01 in an early run on the same topic. **LSA** (truncated SVD to 10 dimensions, then length-normalized) keeps the main directions of variation and raised
it to about 0.3.

**4. Clustering.** **K-Means** groups the vectors. When you do not pass `--k`, it tries 2 to 8 clusters and keeps the one
with the best **silhouette score** (how much closer a paper is to its own group than to the nearest other group, measured
with cosine distance). Each theme is named by the words with the highest average TF-IDF weight in it, and its papers are
sorted by closeness to the theme centre, so the first ones listed are the most representative.

**5. Agents** (`research_assistant/crew.py`, CrewAI). A **researcher** has one tool, `arxiv_cluster_search`, which runs steps
1 to 4 and returns each theme's key terms, size and representative papers as text. A **writer** receives that survey and
writes one section per theme plus a References list; it is told to use only papers it was given.

**6. Front ends.** `cli.py` prints the themes (and optionally the agent summary). `app.py` is a Streamlit page with a theme
scatter plot (2-D SVD coordinates) and an expandable paper list per theme.

## Example run
`python -m research_assistant.cli "retrieval augmented generation" --papers 100` (a live run on 2026-11-07):

```
retrieved 100 unique papers
100 papers grouped into 8 themes (silhouette 0.27)
```
| Theme | Top terms | Papers |
|---|---|---|
| 1 | agent, llm, research, evaluation | 19 |
| 2 | graph, knowledge, kg, multi | 14 |
| 3 | performance, chunking, context, chunks | 14 |
| 4 | models, llms, using, knowledge | 14 |
| 5 | llms, knowledge, generation process, information | 12 |
| 6 | multimodal, visual, video, content | 11 |
| 7 | recommendation, conversational, data, development | 9 |
| 8 | similarity, search, similarity search, proprietary | 7 |

With `--summarize` (`openai/gpt-oss-120b` on Groq) the writer agent produced a markdown summary with one section per theme
and a numbered reference list with arXiv links.

An earlier run sorted newest-first instead of by relevance and mixed in unrelated papers (a music-genre classifier appeared
under "RAG"); sorting by relevance fixed most of that.

## Run it
```bash
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m research_assistant.cli "retrieval augmented generation" --papers 100          # retrieval + clustering, no LLM
python -m research_assistant.cli "graph neural networks" --papers 60 --k 5 --json themes.json   # fixed k, save the clusters

cp .env.example .env     # add a key, then:
python -m research_assistant.cli "retrieval augmented generation" --papers 100 --summarize
streamlit run app.py
```

## Configuration
| Variable | Meaning |
|---|---|
| `RESEARCH_LLM_MODEL` | `groq/<model>` (Groq's OpenAI-compatible endpoint; needs `GROQ_API_KEY`) or any provider CrewAI supports natively, e.g. `gemini/gemini-2.5-flash-lite` with `GEMINI_API_KEY`. |
| `--papers N` | How many papers to retrieve (default 100). |
| `--k N` | Fix the number of themes instead of choosing by silhouette. |

## Tests
`pytest -q` runs 12 offline tests using fake arXiv responses and synthetic abstracts, so no network or key is needed. They
check the query forms, feed parsing, paging (two requests and one pause for 100 papers), stopping when arXiv has fewer papers,
de-duplication across pages, the fall-back to the all-words query, that two synthetic topics separate into two pure clusters,
that the number of clusters is chosen automatically, that every paper lands in exactly one cluster, and the overview text.

## Layout
```
research_assistant/arxiv_client.py   paginated, polite arXiv retrieval
research_assistant/clustering.py     TF-IDF, LSA, K-Means, silhouette, theme naming, overview text
research_assistant/crew.py           CrewAI tool, researcher and writer agents, LLM selection
research_assistant/cli.py            command line
app.py                               Streamlit front end
tests/                               offline tests
```

## Limitations
- Silhouette is a rough guide. On one narrow topic the themes overlap, the score stays low (0.3 or less), and it tends to
  rise with k, so treat the number of themes as adjustable (`--k`). The theme labels are keywords, not written names.
- Clusters are built from abstracts only, not full text. arXiv search is keyword based, so a few loosely related papers get in.
- The writer agent can over-interpret a paper that does not really belong to its theme. Check the references.
- arXiv rate limits apply, and results change over time, so a rerun will not give identical themes.
