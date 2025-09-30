from crewai import Crew, Process
from tasks import research_task, write_task
from agents import topic_researcher, topic_writer, llm


crew = Crew(
    agents=[topic_researcher, topic_writer],
    tasks=[research_task, write_task],
    process=Process.sequential
)

