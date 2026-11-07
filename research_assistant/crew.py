"""Two CrewAI agents: a researcher who retrieves and clusters arXiv papers, and a writer who turns the themes into a summary."""
import os
from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import BaseTool

from research_assistant.arxiv_client import fetch_papers
from research_assistant.clustering import cluster_papers, render_overview

DEFAULT_MODEL = "gemini/gemini-2.5-flash-lite"          # or groq/openai/gpt-oss-120b, openai/gpt-4o-mini, anthropic/claude-...
GROQ_URL = "https://api.groq.com/openai/v1"


def build_llm(model: str | None = None) -> LLM:
    """`groq/<model>` goes through Groq's OpenAI-compatible endpoint, so no LiteLLM install is needed."""
    model = model or os.getenv("RESEARCH_LLM_MODEL", DEFAULT_MODEL)
    if model.startswith("groq/"):
        return LLM(model="openai/" + model[len("groq/"):], base_url=GROQ_URL, api_key=os.environ["GROQ_API_KEY"], temperature=0.3)
    return LLM(model=model, temperature=0.3)


class ArxivClusterTool(BaseTool):
    name: str = "arxiv_cluster_search"
    description: str = (
        "Fetch many recent arXiv papers for a topic (default 100) and group them into themes. "
        "Input: the topic as a short keyword phrase. Returns each theme's key terms, size and representative papers."
    )
    max_papers: int = 100

    def _run(self, topic: str) -> str:
        try:
            papers = fetch_papers(topic, self.max_papers)
            if len(papers) < 4:
                return f"Only {len(papers)} papers found for '{topic}'. Try broader keywords."
            return render_overview(cluster_papers(papers))
        except Exception as exc:                                  # the agent should see the problem and adapt
            return f"Error fetching or clustering arXiv papers: {exc}"


def build_crew(max_papers: int = 100, model: str | None = None, output_file: str = "research_summary.md") -> Crew:
    llm = build_llm(model)
    researcher = Agent(
        role="Research Assistant",
        goal="Survey the research landscape for {topic} by retrieving many papers and identifying the main themes.",
        backstory="You search arXiv at scale and organise papers into themes before reading any single one.",
        tools=[ArxivClusterTool(max_papers=max_papers)], llm=llm, allow_delegation=False, verbose=False,
    )
    writer = Agent(
        role="Research Summarizing Writer",
        goal="Write a structured, referenced summary of {topic} from the researcher's themes.",
        backstory="You write clear research summaries and cite only papers you were given.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    survey = Task(
        description=("Use the arxiv_cluster_search tool once with a short keyword phrase for {topic}. "
                     "Report the themes, how many papers each holds, and the representative papers exactly as returned."),
        expected_output="The theme overview with paper titles and links, unchanged.",
        agent=researcher,
    )
    write = Task(
        description=("Write a markdown summary of {topic}: one short section per theme (what that group of papers is about), "
                     "then a References list. Use only papers from the survey; do not invent papers or findings."),
        expected_output="A markdown summary with one section per theme and a References list.",
        agent=writer, context=[survey], output_file=output_file,
    )
    return Crew(agents=[researcher, writer], tasks=[survey, write], process=Process.sequential)


def run(topic: str, max_papers: int = 100, model: str | None = None, out_dir: str = ".") -> str:
    out = Path(out_dir) / "research_summary.md"
    build_crew(max_papers, model, str(out)).kickoff(inputs={"topic": topic})
    return out.read_text(encoding="utf-8")
