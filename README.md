# AIDA - Atida Intelligent Data Assistant
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aida-demo.streamlit.app/)

AIDA is a specialized AI assistant for Atida pharmaceutical company, built using OpenAI's [Assistants API](https://platform.openai.com/docs/assistants/overview) with [Code Interpreter](https://platform.openai.com/docs/assistants/tools/code-interpreter). The assistant's analysis, including data processing, SQL queries, and visualizations, will be streamed to the app's user interface.

## Current Features

### Data Analysis
- Natural language to SQL query conversion
- BigQuery integration with dataset explorer
- File upload support (CSV, XLSX, JSON)
- Real-time data processing and analysis
- Interactive data visualization with Plotly
- Downloadable visualization exports

### User Interface
- Clean, modern interface with Atida branding
- Expandable code and output sections
- Interactive data tables with sorting and filtering
- File management with size indicators
- Dataset explorer with schema preview

### Chat Experience
- Real-time streaming responses
- Code execution with live output
- Automatic error handling and recovery
- Context-aware responses
- Support for multiple data sources simultaneously
- Persistent chat history with automatic saving
- Chat thread management (create, load, delete)

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

### Authentication & Security
- [ ] Add user authentication system
- [ ] Implement role-based access control
- [x] Add session management
- [ ] Secure file upload handling

### Chat Interface
- [ ] Tables should not disappear from the chat while a message is being typed.
- [ ] Add stop/cancel button for ongoing chat responses
- [ ] Implement up arrow functionality to reload last message
- [ ] Add message editing capabilities
- [ ] Improve error message display

### Visualization & Data
- [ ] Enhanced graph customization options
- [ ] Support for more file formats
- [ ] Batch data processing
- [ ] Export functionality for analysis results

### Performance
- [ ] Implement response caching
- [ ] Optimize large dataset handling
- [ ] Add progress indicators for long operations
- [ ] Improve error recovery mechanisms
