from crewai import Agent
from crewai import LLM

from dotenv import load_dotenv
load_dotenv()

from tools import arxiv_search_tool

import os
from langchain_google_genai import GoogleGenerativeAI
#calling the gemini model
llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    verbose=True,
    temperature=0.3,
    gemini_api_key=os.getenv("GEMINI_API_KEY")
)




#agents

#creating a research assistant agent
topic_researcher = Agent(
    role="Research Assistant",
    goal="Research about the given {topic} from the research papers.",
    verbose=True,
    memory=True,
    backstory="You are a research assistant who is an expert in finding relevant information from research papers.",
    tools=[arxiv_search_tool],
    llm=llm,
    allow_delegation=True
)

#creating a research summarizing writer agent
topic_writer = Agent(
    role="Research Summarizing Writer",
    goal="Write a detailed research summary about the given {topic} in a well-structured format including the references.",
    verbose=True,
    memory=True,
    backstory="You are a research writer who is an expert in writing detailed research summaries.",
    tools=[arxiv_search_tool],
    llm=llm,
    allow_delegation=False
)

