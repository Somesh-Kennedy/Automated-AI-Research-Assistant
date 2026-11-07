# Automated AI Research Assistant

Give it a research topic. It pulls **~100 papers from arXiv**, **clusters them into themes**, and two CrewAI agents turn
those themes into a structured, referenced summary.

```
topic ─► arXiv API (paginated, 50 per request, 3 s apart) ─► TF-IDF ─► LSA (10 dims) ─► K-Means (k by silhouette)
      ─► researcher agent reads the theme overview ─► writer agent drafts the markdown summary + references
```

## What each part does
- `research_assistant/arxiv_client.py` fetches up to N unique papers: exact-phrase query first (most relevant first), falling
  back to "all words must appear" if the phrase is rare. De-duplicates across pages and respects arXiv's request spacing.
- `research_assistant/clustering.py` groups papers into themes. Title and abstract go through TF-IDF, then LSA to remove
  noise, then K-Means. The number of themes is picked by silhouette score (2 to 8) unless you pass `--k`. Each theme gets
  its top terms and its most central papers first.
- `research_assistant/crew.py` is the CrewAI part: a **researcher** with one tool (fetch + cluster) and a **writer** that may
  cite only papers it was given.
- `app.py` is a Streamlit front end with a theme scatter plot and a paper list per theme.

## Run it
```bash
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m research_assistant.cli "retrieval augmented generation" --papers 100          # retrieval + clustering, no LLM
cp .env.example .env     # add a key, then:
python -m research_assistant.cli "retrieval augmented generation" --papers 100 --summarize
streamlit run app.py
```
`RESEARCH_LLM_MODEL` can be `groq/<model>` (uses Groq's OpenAI-compatible endpoint, needs `GROQ_API_KEY`) or any provider
CrewAI supports natively, for example `gemini/gemini-2.5-flash-lite` with `GEMINI_API_KEY`.

## Verified behaviour
- 100 unique papers retrieved for "retrieval augmented generation", clustered into 8 themes (silhouette about 0.3). The themes
  were readable: multi-hop and graph RAG, chunking and context, multimodal RAG, recommendation, similarity search, and others.
- The full agent run (`openai/gpt-oss-120b` on Groq) produced a referenced markdown summary with one section per theme.
- `pytest -q`: 12 offline tests (fake arXiv responses and synthetic abstracts; no network or key needed).

## Limitations
- Silhouette is a rough guide. On one narrow topic the themes overlap, the score is low (0.3 or less) and it tends to rise
  with k, so treat the number of themes as adjustable (`--k`).
- Clusters are built from abstracts only, not full text. arXiv search is keyword based, so a few loosely related papers get in.
- The writer agent can over-interpret a paper that does not really belong to its theme. Check the references.
- arXiv rate limits apply; 100 papers take about two requests plus a pause.
