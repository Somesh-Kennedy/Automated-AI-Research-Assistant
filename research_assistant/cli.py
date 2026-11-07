"""python -m research_assistant.cli "graph neural networks" --papers 100 [--summarize]"""
import argparse
import json

from research_assistant.arxiv_client import fetch_papers
from research_assistant.clustering import cluster_papers, render_overview


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve arXiv papers for a topic and cluster them into themes.")
    ap.add_argument("topic")
    ap.add_argument("--papers", type=int, default=100)
    ap.add_argument("--k", type=int, default=None, help="fix the number of clusters (default: chosen by silhouette)")
    ap.add_argument("--json", help="also write the clusters to this file")
    ap.add_argument("--summarize", action="store_true", help="run the CrewAI agents and write research_summary.md")
    args = ap.parse_args()

    papers = fetch_papers(args.topic, args.papers)
    print(f"retrieved {len(papers)} unique papers")
    result = cluster_papers(papers, k=args.k)
    print(render_overview(result))
    if args.json:
        data = [{"theme": c.top_terms, "size": c.size, "papers": [p.__dict__ for p in c.papers]} for c in result.clusters]
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"silhouette": result.silhouette, "clusters": data}, f, indent=2)
    if args.summarize:
        from research_assistant.crew import run
        print(run(args.topic, args.papers))


if __name__ == "__main__":
    main()
