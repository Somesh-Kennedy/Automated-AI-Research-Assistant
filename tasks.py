from crewai import Task
from agents import topic_researcher, topic_writer
from tools import arxiv_search_tool

research_task = Task(
    description=(
        "Identify the research topic related papers and delegate the research to the Research Assistant agent."
        "Focus on identifying the process, methodology, reasons and results from the related research papers."
        "Your final report should be a well-structured research summary."
    ),
    expected_output='A comprehensive research summary of 3 paragraphs on the given topic including references below it.',
    tools=[arxiv_search_tool],
    agent=topic_researcher,
)

write_task = Task(
    description=(
        "Write a detailed research summary based on the research findings provided by the Research Assistant agent."
        "Your summary should be well-structured and cover all key aspects of the research."
    ),
    expected_output='A 5 paragraph detailed research summary on {topic} formatted as markdown including references below it.',
    tools=[arxiv_search_tool],
    agent=topic_writer,
    async_execution=False,
    output_file="research_summary1.md"
)
