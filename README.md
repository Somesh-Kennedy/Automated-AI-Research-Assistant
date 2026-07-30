# Automated AI Research Assistant

This project is a Streamlit application that uses **CrewAI** to automate the process of researching a given topic, generating a structured summary based on **arXiv** research articles, and allowing users to ask follow-up questions on the generated summary.

It leverages the power of Large Language Models (LLMs) via the `langchain-google-genai` library (specifically with **Gemini**) and custom tools to create a collaborative team of AI agents for research and summarization.

---

## Features

* **Automated Research:** A multi-agent crew (Research Assistant and Research Summarizing Writer) works together to research a topic.
* **arXiv Integration:** A custom `arxiv_search` tool fetches relevant, up-to-date research papers.
* **Structured Summaries:** Generates detailed, multi-paragraph research summaries, complete with a references section.
* **Interactive Q&A:** Allows users to ask follow-up questions directly on the generated summary.
* **Downloadable Output:** The research summary is available for download as a markdown file.
* **Streamlit Interface:** A clean and interactive web interface for a smooth user experience.

---

## Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.8+**
2.  A **Gemini API Key**.

---

## Getting Started

### 1. Setup Environment

Clone the repository (assuming you have your project files in a single directory):
```bash
# Clone your project repository (if applicable)
# git clone <your-repo-link>
# cd automated-ai-research-assistant
```

### 2. Install Dependencies

Install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Configure API Key

The project requires a Gemini API key for the LLM.

1.  Open the **`.env`** file.

2.  Replace the placeholder with your actual key:
```ini
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    CREWAI_TRACING_ENABLED=true
```

    > **Note:** The `llm` is configured in `agents.py` to use `gemini-2.5-flash-lite`.

### 4. Run the Application

Start the Streamlit application from your terminal:
```bash
streamlit run app.py
```

This will open the application in your web browser.

---

## How to Use

1.  **Enter Topic:** In the sidebar, enter a research topic (e.g., "Augmented Reality in Healthcare").
2.  **Run Research:** Click the **"Run Research"** button. The CrewAI process will start, and a loading spinner will appear.
3.  **View Summary:** Once complete, the detailed research summary will appear in the main column.
4.  **Ask Questions:** Use the **"Ask Questions"** section to query the summary with follow-up questions.
5.  **Download:** Click **"Download Summary"** to save the output locally.
6.  **Clear:** Click **"Clear Summary"** to start a new research session.

---

## Project Structure

| File | Description |
| :--- | :--- |
| `app.py` | The main Streamlit application, handling the UI and `crew.kickoff`. |
| `crew.py` | Defines the CrewAI **Crew** with the agents and tasks. |
| `agents.py` | Defines the LLM and the two CrewAI **Agents** (`topic_researcher`, `topic_writer`). |
| `tasks.py` | Defines the CrewAI **Tasks** for research and writing, including the expected output and file saving. |
| `tools.py` | Contains the custom **`ArxivSearchTool`** for fetching papers from arXiv. |
| `requirements.txt` | Lists all necessary Python dependencies. |
| `style.css` | Custom CSS for the Streamlit application's styling. |
| `.env` | Environment file for storing the `GEMINI_API_KEY`. |

---

## CrewAI Agents and Tasks

### Agents

* **Research Assistant (topic_researcher):**
    * **Role:** Expert in finding relevant information from research papers.
    * **Goal:** Research the given topic from research papers, focusing on process, methodology, reasons, and results.
    * **Tools:** `arxiv_search_tool`

* **Research Summarizing Writer (topic_writer):**
    * **Role:** Expert in writing detailed, well-structured research summaries.
    * **Goal:** Write a detailed research summary about the topic, including references.
    * **Tools:** `arxiv_search_tool` (Used for referencing/validation)

### Tasks

1.  **Research Task:** Delegates the search to the Research Assistant.
2.  **Write Task:** Delegates the writing to the Research Summarizing Writer.
    * **Expected Output:** A 5-paragraph detailed research summary formatted as markdown, including a references section.
    * **Output File:** `research_summary1.md`

The crew is configured to run in **`Process.sequential`** mode, meaning the Research Assistant performs its task first, and then the Research Summarizing Writer uses those findings to complete the writing task.
