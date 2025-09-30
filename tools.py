from crewai.tools import BaseTool
import requests
import feedparser

class ArxivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = (
        "Search for research papers on arXiv based on the topic. "
        "Input should be a keyword or query string. "
        "Returns a list of paper titles, authors, published date, summary and links."
    )

    def _run(self, query: str) -> str:
        try:
            url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
            response = requests.get(url)
            feed = feedparser.parse(response.text)

            results = []
            for entry in feed.entries:
                paper = {
                    "title": entry.title,
                    "authors": [author.name for author in entry.authors],
                    "published": entry.published,
                    "link": entry.link,
                    "summary": entry.summary[:300] + "..."  # truncate for brevity
                }
                results.append(paper)

            
            output = "\n\n".join([
                f" {p['title']}\n Authors: {', '.join(p['authors'])}\n Published: {p['published']}\n {p['link']}\n {p['summary']}"
                for p in results
            ])
            return output

        except Exception as e:
            return f"Error fetching from arXiv: {str(e)}"



arxiv_search_tool = ArxivSearchTool()