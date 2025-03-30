# AIDA - Atida Intelligent Data Assistant
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aida-demo.streamlit.app/)

AIDA is a specialized AI assistant for Atida pharmaceutical company, built using OpenAI's [Assistants API](https://platform.openai.com/docs/assistants/overview) with [Code Interpreter](https://platform.openai.com/docs/assistants/tools/code-interpreter). The assistant's analysis, including data processing, SQL queries, and visualizations, will be streamed to the app's user interface.

## Happy Path Demo

### Path
1. User logs in
2. Retrieve previous conversation
3. User selects dataset
4. User asks questions
5. User receives answers

- Link to the demo chatgpt link (https://chatgpt.com/)

## Current Features

### Data Analysis
- Natural language to SQL query conversion
- BigQuery integration with dataset explorer
- Configurable query row limit (1-100,000 rows)
- File upload support (CSV, XLSX, JSON)
- Real-time data processing and analysis
- Interactive data visualization with Plotly
- Downloadable visualization exports
- Automatic context inclusion of all data sources
- Query result size control with adjustable limits

### User Interface
- Clean, modern interface with Atida branding
- Expandable code and output sections
- Interactive data tables with sorting and filtering
- File management with size indicators
- Simplified data source management
- Dataset explorer with schema preview
- Query limit control in sidebar
- Improved DataFrame display handling

### Chat Experience
- Real-time streaming responses
- Code execution with live output
- Automatic error handling and recovery
- Context-aware responses
- Support for multiple data sources simultaneously
- Persistent chat history with text and code only
- Chat thread management (create, load, delete)
- Optimized message storage for better performance

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

## TODO

### User Experience
- [x] Remove the table schema information from sidebar
- [x] Add query size control
- [ ] Add stop/cancel button for ongoing chat responses
- [ ] Improve error message display

### Chat Response
- [x] Optimize DataFrame handling
- [x] Implement query size limits
- [ ] Adjust assistant
- [ ] Save the last query response in a variable for future use
- [ ] Allow to code interpreter execute queries
- [ ] Get tables description from BigQuery to pass to context

### Bug Fixes
- [ ] Fix the issue with images and graphs not displaying in the chat
- [x] Keep historical chat data
- [ ] Always show tabular data in the data table
- [x] Code interpreter tab closed by default
- [x] Handle large query results efficiently

### Infrastructure
- [ ] Evaluate migration from BigQuery to Druid
  - Pros:
    - Better performance for real-time analytics
    - Optimized for aggregations and time-series
    - Lower query latency
    - More predictable costs (no pay-per-query/scanned data)
    - Excellent for timestamped events data
  - Cons:
    - More complex to maintain than BigQuery (managed service)
    - Requires infrastructure management
    - Less flexible for complex ad-hoc queries
    - Limited JOIN capabilities
    - Needs cluster sizing and capacity planning
  - Decision factors:
    - Stay with BigQuery if:
      - Complex queries with multiple JOINs are common
      - Cost is not a major concern
      - Full SQL flexibility is needed
      - No team available for infrastructure maintenance
    - Move to Druid if:
      - Queries are mainly aggregations/metrics
      - Millisecond response times required
      - Mostly time-series data
      - Team available for infrastructure maintenance
      - BigQuery costs are a concern






