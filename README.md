# AIDA - Atida Intelligent Data Assistant
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aida-demo.streamlit.app/)

AIDA is a specialized AI assistant for Atida pharmaceutical company, built using OpenAI's [Assistants API](https://platform.openai.com/docs/assistants/overview) with [Code Interpreter](https://platform.openai.com/docs/assistants/tools/code-interpreter). The assistant's analysis, including data processing, SQL queries, and visualizations, will be streamed to the app's user interface.

Key capabilities:
- Analyzing pharmaceutical data
- Creating visualizations
- Executing SQL queries
- Searching and analyzing files

## Quick Start

1. Clone this repository
2. Install the required dependencies by running

```python
pip install -r requirements.txt
```
   
3. Modify `create_assistant.py` as needed, and note down the `ASSISTANT_ID`.
4. Create a `secrets.toml` file located within the `.streamlit/` directory. It should minimally contain these variables: `OPENAI_API_KEY`, `ASSISTANT_ID`
5. Launch the application:

```python
streamlit run chat_app.py
```
