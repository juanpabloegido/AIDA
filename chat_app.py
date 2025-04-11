import base64
import os
import time
from datetime import datetime
import logging
import json
from contextlib import redirect_stdout
import io

import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from google.oauth2 import service_account
from openai.types.beta.assistant_stream_event import (
    ThreadRunStepCreated,
    ThreadRunStepDelta,
    ThreadRunStepCompleted,
    ThreadMessageCreated,
    ThreadMessageDelta
)
from openai.types.beta.threads.text_delta_block import TextDeltaBlock
from openai.types.beta.threads.runs.tool_calls_step_details import ToolCallsStepDetails
from openai.types.beta.threads.runs.code_interpreter_tool_call import (
    CodeInterpreterOutputImage,
    CodeInterpreterOutputLogs
)
from auth import login, logout, save_chat_history, load_chat_history, get_chat_thread, delete_chat_thread

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIDA")

# Set page configuration
st.set_page_config(
    page_title="AIDA - Atida Intelligent Data Assistant",
    page_icon="💊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTitle {
        color: #2e4d7b;
    }
    #MainMenu {visibility: hidden}
    #header {visibility: hidden}
    #footer {visibility: hidden}
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .chat-thread {
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
    }
    .chat-thread:hover {
        background-color: #e6e6e6;
    }
    </style>
""", unsafe_allow_html=True)

# Check authentication
if not login():
    st.stop()

# Initialize clients and credentials
@st.cache_resource
def init_clients():
    # OpenAI setup
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    assistant = client.beta.assistants.retrieve(st.secrets["ASSISTANT_ID"])
    return client, assistant

# Initialize clients
openai_client, assistant = init_clients()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hi {st.session_state.user.email}! I'm AIDA, your intelligent data assistant. How can I help you today? 👋"}]
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "thread_id" not in st.session_state:
    # Create initial thread
    try:
        thread = openai_client.beta.threads.create()
        st.session_state.thread_id = thread.id
        logger.info(f"Created initial thread: {thread.id}")
    except Exception as e:
        logger.error(f"Error creating initial thread: {str(e)}")
        st.error("Error initializing the assistant. Please refresh the page.")
        st.stop()
if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = []

# BigQuery Explorer Class
class BigQueryExplorer:
    def __init__(self, bq_client):
        self.client = bq_client
        self._cache = {}
        self.debug_info = []

    def log_debug(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.debug_info.append(f"[{timestamp}] {message}")
        logger.info(message)

    def get_datasets(self):
        if 'datasets' not in self._cache:
            try:
                self._cache['datasets'] = list(self.client.list_datasets())
            except Exception as e:
                st.error(f"Error fetching datasets: {str(e)}")
                return []
        return self._cache['datasets']

    def get_tables(self, dataset_id):
        cache_key = f'tables_{dataset_id}'
        if cache_key not in self._cache:
            try:
                self._cache[cache_key] = list(self.client.list_tables(dataset_id))
            except Exception as e:
                st.error(f"Error fetching tables: {str(e)}")
                return []
        return self._cache[cache_key]

    def get_table_schema(self, dataset_id, table_id):
        cache_key = f'schema_{dataset_id}_{table_id}'
        if cache_key not in self._cache:
            try:
                table = self.client.get_table(f"{dataset_id}.{table_id}")
                self._cache[cache_key] = {
                    'schema': table.schema,
                    'size_mb': table.num_bytes / 1024 / 1024,
                    'description': table.description or 'No description available'
                }
            except Exception as e:
                st.error(f"Error fetching schema: {str(e)}")
                return None
        return self._cache[cache_key]

    def display_explorer(self):
        st.sidebar.markdown("### 📊 Dataset Explorer")
        datasets = self.get_datasets()

        if not datasets:
            st.sidebar.warning("No datasets available")
            return

        selected_dataset = st.sidebar.selectbox(
            "Select dataset:",
            options=[d.dataset_id for d in datasets]
        )

        if selected_dataset:
            tables = self.get_tables(selected_dataset)
            for table in tables:
                with st.sidebar.expander(f"📋 {table.table_id}"):
                    schema = self.get_table_schema(selected_dataset, table.table_id)
                    if schema:
                        st.markdown(f"**Description:** {schema['description']}")
                        st.markdown(f"**Size:** {schema['size_mb']:.2f} MB")
                        st.markdown("**Columns:**")
                        for field in schema['schema']:
                            st.markdown(f"- {field.name} ({field.field_type})")


# Sidebar
with st.sidebar:
    st.image("https://www.atida.com/static/version1741757720/frontend/Interactiv4/mifarmaHyva/es_ES/images/logo.svg",
             width=200)
    logout()
    st.divider()

    # Chat History Section
    st.markdown("### 💬 Chat History")
    
    # New Chat button
    if st.button("🆕 New Chat", type="primary", use_container_width=True):
        # Create new thread
        try:
            thread = openai_client.beta.threads.create()
            st.session_state.thread_id = thread.id
            logger.info(f"Created new thread: {thread.id}")
            st.session_state.messages = [
                {"role": "assistant", "content": f"Hi {st.session_state.user.email}! I'm AIDA, your intelligent data assistant. How can I help you today? 👋"}]
            st.rerun()
        except Exception as e:
            logger.error(f"Error creating new thread: {str(e)}")
            st.error("Error creating new chat. Please try again.")
    
    # Load chat history
    chat_history = load_chat_history(st.session_state.user.id)
    
    for chat in chat_history:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"📝 {chat['title']}", key=f"chat_{chat['thread_id']}", use_container_width=True):
                st.session_state.messages = chat['messages']
                st.session_state.thread_id = chat['thread_id']
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{chat['thread_id']}"):
                if delete_chat_thread(chat['thread_id']):
                    st.toast("Chat deleted successfully!")
                    if st.session_state.thread_id == chat['thread_id']:
                        st.session_state.messages = [
                            {"role": "assistant", "content": f"Hi {st.session_state.user.email}! I'm AIDA, your intelligent data assistant. How can I help you today? 👋"}]
                        st.session_state.thread_id = None
                    st.rerun()
    
    st.divider()

    # Data Sources Section
    st.markdown("### 📁 Data Sources")

    # Query Limit Control
    query_limit = st.number_input(
        "Max rows per query",
        min_value=1,
        max_value=100000,
        value=st.session_state.get('query_limit', 1000),
        help="Maximum number of rows to return in query results"
    )
    st.session_state.query_limit = query_limit

    # File Upload Section
    uploaded_files = st.file_uploader(
        "Upload dataset(s)",
        accept_multiple_files=True,
        type=["csv", "xlsx", "json"]
    )

    if st.button("Process Files", type="primary"):
        with st.spinner("Processing files..."):
            file_ids = []
            for file in uploaded_files:
                try:
                    oai_file = openai_client.files.create(
                        file=file,
                        purpose='assistants'
                    )
                    file_ids.append(oai_file.id)

                    st.session_state.uploaded_files_info.append({
                        "name": file.name,
                        "type": file.type,
                        "size": file.size,
                        "file_id": oai_file.id
                    })
                    logger.info(f"Uploaded file: {file.name} with ID: {oai_file.id}")
                except Exception as e:
                    st.error(f"Error uploading {file.name}: {str(e)}")
                    continue

            if file_ids:
                try:
                    thread = openai_client.beta.threads.create()
                    st.session_state.thread_id = thread.id

                    openai_client.beta.threads.update(
                        thread_id=thread.id,
                        tool_resources={"code_interpreter": {"file_ids": file_ids}}
                    )

                    st.session_state.file_uploaded = True
                    st.toast("✅ Files processed successfully!", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error setting up thread: {str(e)}")

    # Show uploaded files
    if st.session_state.uploaded_files_info:
        st.markdown("#### 📄 Uploaded Files")
        for idx, file_info in enumerate(st.session_state.uploaded_files_info):
            col1, col2, col3 = st.columns([0.6, 0.3, 0.1])
            with col1:
                st.text(file_info['name'])
            with col2:
                st.text(f"{file_info['size'] / 1024:.1f} KB")
            with col3:
                if st.button("🗑️", key=f"delete_{idx}"):
                    try:
                        openai_client.files.delete(file_info['file_id'])
                        st.session_state.uploaded_files_info.pop(idx)
                        st.toast(f"Deleted {file_info['name']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {str(e)}")

    # BigQuery Tables Section
    bq_client = bigquery.Client(credentials=service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/bigquery"]
    ))
    bq_explorer = BigQueryExplorer(bq_client)
    datasets = bq_explorer.get_datasets()
    if datasets:
        st.markdown("#### 📊 Datamars")
        for dataset in datasets:
            with st.expander(f"📁 {dataset.dataset_id}"):
                tables = bq_explorer.get_tables(dataset.dataset_id)
                for table in tables:
                    st.text(table.table_id)

    st.divider()
    st.caption("Powered by Atida © 2025")

# Main chat interface
st.title("💊 AIDA - Atida Intelligent Data Assistant")
st.caption("Your smart partner in pharmaceutical data insights")

# Add Tips section
with st.expander("💡 Tips", expanded=False):
    st.markdown("""
    ### How to use AIDA
    
    **Ask questions about your data:**
    - Query your data with natural language: "Show me sales trends for the last quarter"
    - Write SQL directly: "SELECT * FROM dataset.table WHERE date > '2023-01-01'"
    
    **Create visualizations:**
    - Ask for plots using natural language: "Create a bar chart of sales by region"
    - Request specific chart types: "Plot a histogram of patient ages"
    - Visualize query results: "Show me a pie chart of market share by product"
    
    **Data analysis:**
    - Get statistical insights: "Calculate summary statistics for this dataset"
    - Find patterns: "Identify trends in prescription data over time"
    - Compare data: "Compare sales performance across different regions"
    
    **Pro tip:** Upload your CSV/Excel files or connect to BigQuery to analyze your data.
    """)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item["type"] == "text":
                    st.markdown(item["content"])
                elif item["type"] == "image":
                    for image_html in item["content"]:
                        st.html(image_html)
                elif item["type"] == "code":
                    with st.status(item["label"], state="complete"):
                        st.code(item["content"])
                elif item["type"] == "dataframe":
                    # Reconstruct DataFrame from serialized format
                    df_dict = item["content"]
                    df = pd.DataFrame(df_dict["records"], columns=df_dict["columns"])
                    # Convert back to original dtypes if available
                    if "dtypes" in df_dict:
                        for col, dtype in df_dict["dtypes"].items():
                            try:
                                df[col] = df[col].astype(dtype)
                            except:
                                pass  # Keep original dtype if conversion fails
                    st.dataframe(df, use_container_width=True)
        else:
            st.markdown(message["content"])


# Function to get context for the assistant
def get_assistant_context():
    context = []

    # Add all files context
    if st.session_state.uploaded_files_info:
        file_info = []
        for info in st.session_state.uploaded_files_info:
            file_info.append(
                f"- {info['name']} (ID: {info['file_id']}, Type: {info['type']})\n"
                f"  Access path: /mnt/data/{info['file_id']}"
            )
        if file_info:
            context.append("Available Files:\n" + "\n".join(file_info))

    # Add all tables context
    datasets = bq_explorer.get_datasets()
    if datasets:
        tables_info = []
        for dataset in datasets:
            tables = bq_explorer.get_tables(dataset.dataset_id)
            for table in tables:
                table_id = f"{dataset.dataset_id}.{table.table_id}"
                schema = bq_explorer.get_table_schema(dataset.dataset_id, table.table_id)
                if schema:
                    fields_info = []
                    for field in schema['schema']:
                        desc = field.description or 'No description'
                        fields_info.append(f"{field.name} ({field.field_type}): {desc}")
                    
                    tables_info.append(
                        f"Table: `{table_id}`\n"
                        f"Description: {schema['description']}\n"
                        f"Columns:\n" + "\n".join(f"- {field}" for field in fields_info) + f"\n"
                    )
        if tables_info:
            context.append("Available BigQuery Tables:\n" + "\n\n".join(tables_info))

    return "\n\n".join(context)


# Function to create visualization based on dataframe
def display_dataframe(df, title="Results", query=None):
    """Display a dataframe in a consistent format"""
    if isinstance(df, str):
        st.error(df)
        return

    try:
        content = []
        
        # Add title and query as text content
        text_content = f"#### {title}\n"
        if query:
            text_content += f"**Query:**\n```sql\n{query}\n```\n"
        content.append({"type": "text", "content": text_content})
        
        # Display DataFrame in UI
        st.markdown(text_content)
        st.dataframe(df, use_container_width=True)
        
        # Add DataFrame to content for Streamlit history
        content.append({
            "type": "dataframe", 
            "content": {
                "records": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict()
            }
        })
        
        # Convert to string for AIDA thread, limit the size
        max_rows = min(100, len(df))  # Show at most 100 rows in the thread
        df_preview = df.head(max_rows)
        df_string = df_preview.to_string(index=False)
        if len(df) > max_rows:
            df_string += f"\n\n... and {len(df) - max_rows} more rows"
        
        # Add to thread for AIDA with size limit
        thread_content = f"{text_content}\n```\n{df_string}\n```"
        if len(thread_content) > 200000:  # Safe limit for OpenAI
            thread_content = thread_content[:200000] + "\n...(truncated)"
        
        openai_client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=thread_content
        )
        logger.info(f"DataFrame preview ({max_rows} rows) added to thread as text")
        
        # Add to message history without saving to database
        st.session_state.messages.append({"role": "assistant", "content": content})
        
        # Show success message
        st.caption("✅ Query executed successfully!")

    except Exception as e:
        logger.error(f"Error in display_dataframe: {str(e)}")
        st.error(f"Error displaying dataframe: {str(e)}")
        return None

    return content


# Function to execute BigQuery
def execute_query(query, show_results=True):
    try:
        # Add LIMIT clause if not present
        query = query.strip()
        limit = st.session_state.get('query_limit', 1000)
        
        # Remove any existing LIMIT clause
        query_lower = query.lower()
        if 'limit' in query_lower:
            # Remove everything from LIMIT onwards
            query = query[:query_lower.find('limit')].strip()
        
        # Add our controlled LIMIT
        query = f"{query}\nLIMIT {limit}"
        
        # Execute with dry run first to validate
        job_config = bigquery.QueryJobConfig(dry_run=True)
        bq_client.query(query, job_config=job_config)
        logger.info("SQL query validation passed")
        
        # Execute actual query
        df = bq_client.query(query).to_dataframe()
        if show_results:
            display_dataframe(df, "Query Results", query)
        return df
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None


# Function to ensure we have a valid thread
def ensure_thread():
    """Ensure we have a valid thread or create a new one"""
    try:
        if not hasattr(st.session_state, 'thread_id') or not st.session_state.thread_id:
            logger.info("Creating new thread - no thread_id in session")
            thread = openai_client.beta.threads.create()
            st.session_state.thread_id = thread.id
            logger.info(f"Created new thread: {thread.id}")
            
            # Save initial message
            save_chat_history(
                user_id=None,
                thread_id=thread.id,
                messages=st.session_state.messages,
                title="New Chat"
            )
            return

        # Verify thread exists
        try:
            logger.info(f"Verifying thread: {st.session_state.thread_id}")
            openai_client.beta.threads.retrieve(st.session_state.thread_id)
            logger.info("Thread verified successfully")
        except Exception as e:
            logger.error(f"Thread {st.session_state.thread_id} not found: {str(e)}")
            # Create new thread if current one is invalid
            thread = openai_client.beta.threads.create()
            st.session_state.thread_id = thread.id
            logger.info(f"Created replacement thread: {thread.id}")
            
            # Save initial message
            save_chat_history(
                user_id=None,
                thread_id=thread.id,
                messages=st.session_state.messages,
                title="New Chat"
            )
    except Exception as e:
        logger.error(f"Error in ensure_thread: {str(e)}")
        st.error("Error with assistant thread. Please refresh the page.")
        st.stop()


# Ensure we have a valid thread at startup
ensure_thread()

# Function to convert natural language to SQL
def natural_to_sql(prompt, context):
    """Convert natural language query to SQL using OpenAI"""
    try:
        logger.info(f"Converting natural language to SQL. Prompt: {prompt}")
        logger.info(f"Context provided: {context}")

        # Create a system message with the context
        messages = [
            {
                "role": "system", 
                "content": f"""You are a SQL expert that converts natural language to SQL queries.
                You have access to the following tables and their schemas:

                {context}

                Convert the user's question to a SQL query. Return ONLY the raw SQL query, no markdown formatting, no backticks.
                The query MUST:
                - Start with SELECT or WITH
                - Use proper BigQuery SQL syntax
                - Use fully qualified table names (dataset.table)
                - Use only columns that exist in the schema
                - Be properly formatted with newlines and indentation
                
                If you cannot convert the question to SQL, return 'CANNOT_CONVERT'."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Get SQL query from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0
        )

        sql_query = response.choices[0].message.content.strip()
        
        # Log the generated query
        logger.info(f"Generated SQL query: {sql_query}")
        
        # Remove markdown SQL formatting if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        # Validate basic SQL structure
        if sql_query == 'CANNOT_CONVERT':
            logger.warning("Could not convert natural language to SQL")
            return None
            
        if not sql_query.lower().strip().startswith(("select", "with")):
            logger.error(f"Generated query doesn't start with SELECT or WITH: {sql_query}")
            return None

        return sql_query

    except Exception as e:
        logger.error(f"Error converting natural language to SQL: {str(e)}")
        return None

# Chat input
if prompt := st.chat_input("Ask me about your data..."):
    # Double check thread validity before proceeding
    ensure_thread()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_content = []
        try:
            # Check if it's a SQL query or needs conversion
            if prompt.lower().strip().startswith(("select", "with")):
                sql_query = prompt
                should_try_sql = True
            else:
                # Get context about selected data sources
                context = get_assistant_context()
                sql_query = natural_to_sql(prompt, context)
                should_try_sql = bool(sql_query)

            # Try SQL execution first if we have a query
            sql_success = False
            if should_try_sql and sql_query:
                try:
                    # Log query before execution
                    logger.info(f"Executing SQL query: {sql_query}")
                    
                    # Try to validate query first
                    try:
                        # Dry run to check syntax
                        job_config = bigquery.QueryJobConfig(dry_run=True)
                        bq_client.query(sql_query, job_config=job_config)
                        logger.info("SQL query validation passed")
                    except Exception as e:
                        logger.error(f"SQL query validation failed: {str(e)}")
                        raise Exception(f"Query validation failed: {str(e)}")

                    # Execute actual query
                    df = bq_client.query(sql_query).to_dataframe()

                    # Show the generated SQL
                    if not prompt.lower().strip().startswith(("select", "with")):
                        st.code(sql_query, language="sql", line_numbers=True)

                    # Display results
                    display_dataframe(df, "Query Results", sql_query)
                    sql_success = True

                    # Create simplified version for storage
                    storage_content = [
                        {"type": "text", "content": f"Query executed:\n```sql\n{sql_query}\n```"}
                    ]

                except Exception as e:
                    logger.error(f"Query execution error: {str(e)}")
                    if prompt.lower().strip().startswith(("select", "with")):
                        # Only show error for direct SQL queries
                        error_msg = f"❌ Error executing query: {str(e)}\n\nSQL Query:\n```sql\n{sql_query}\n```"
                        st.error(error_msg)
                        response_content = [{"type": "text", "content": error_msg}]
                        storage_content = response_content
                        sql_success = False
                    else:
                        # For natural language queries, fallback to assistant
                        logger.info("SQL execution failed, falling back to assistant")
                        sql_success = False

            # If SQL failed or wasn't attempted, use the assistant
            if not sql_success:
                # Get context about selected data sources
                context = get_assistant_context()

                # Create message with context
                openai_client.beta.threads.messages.create(
                    thread_id=st.session_state.thread_id,
                    role="user",
                    content=f"{context}\n\nUser question: {prompt}"
                )

                # Create and stream run
                stream = openai_client.beta.threads.runs.create(
                    thread_id=st.session_state.thread_id,
                    assistant_id=assistant.id,
                    stream=True
                )

                message_placeholder = st.empty()
                code_placeholder = st.empty()
                result_placeholder = st.empty()

                for event in stream:
                    if isinstance(event, ThreadRunStepCreated):
                        if event.data.step_details.type == "tool_calls":
                            response_content.append({"type": "code", "label": "Code", "content": ""})
                            code_placeholder = st.empty()

                    elif isinstance(event, ThreadRunStepDelta):
                        if event.data.delta.step_details and event.data.delta.step_details.tool_calls:
                            code_interpreter = event.data.delta.step_details.tool_calls[0].code_interpreter
                            if code_interpreter and code_interpreter.input:
                                response_content[-1]["content"] += code_interpreter.input
                                code_placeholder.code(response_content[-1]["content"], language="python")

                    elif isinstance(event, ThreadMessageCreated):
                        response_content.append({"type": "text", "content": ""})
                        message_placeholder = st.empty()

                    elif isinstance(event, ThreadMessageDelta):
                        if isinstance(event.data.delta.content[0], TextDeltaBlock):
                            text_value = event.data.delta.content[0].text.value
                            if text_value is not None:
                                response_content[-1]["content"] += text_value
                                message_placeholder.markdown(response_content[-1]["content"])

                    elif isinstance(event, ThreadRunStepCompleted):
                        if isinstance(event.data.step_details, ToolCallsStepDetails):
                            code_interpreter = event.data.step_details.tool_calls[0].code_interpreter

                            # Show code being executed
                            if code_interpreter.input:
                                with st.expander("🔍 Code", expanded=False):
                                    st.code(code_interpreter.input, language="python")

                            if code_interpreter.outputs:
                                for output in code_interpreter.outputs:
                                    if isinstance(output, CodeInterpreterOutputImage):
                                        image_data = openai_client.files.content(output.image.file_id).read()
                                        encoded_image = base64.b64encode(image_data).decode()

                                        with st.expander("📊 Visualization", expanded=True):
                                            # Create a download button for the image
                                            st.download_button(
                                                label="Download Visualization",
                                                data=image_data,
                                                file_name="visualization.png",
                                                mime="image/png"
                                            )

                                            # Display the image
                                            image_html = f'<img src="data:image/png;base64,{encoded_image}" style="max-width:100%">'
                                            st.html(image_html)
                                            response_content.append({
                                                "type": "image",
                                                "content": [image_html]
                                            })

                                    elif isinstance(output, CodeInterpreterOutputLogs):
                                        with st.expander("📝 Output", expanded=False):
                                            st.code(output.logs)
                                            response_content.append({
                                                "type": "code",
                                                "label": "Code Output",
                                                "content": output.logs
                                            })

                                            # Try to parse output as DataFrame if it looks like tabular data
                                            try:
                                                if "," in output.logs and "\n" in output.logs:
                                                    df = pd.read_csv(io.StringIO(output.logs))
                                                    display_dataframe(df, "Data Preview", sql_query)
                                            except:
                                                pass  # Not tabular data

                # Create simplified version for storage
                storage_content = []
                for item in response_content:
                    if item["type"] in ["text", "code"]:
                        storage_content.append(item)

            # Update session state with full content for display
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            
            # Save simplified version to database
            storage_messages = []
            for msg in st.session_state.messages:
                if isinstance(msg.get("content"), list):
                    # Filter out dataframes and images, keep only text and code
                    filtered_content = []
                    for item in msg["content"]:
                        if item["type"] in ["text", "code"]:
                            filtered_content.append(item)
                    storage_messages.append({"role": msg["role"], "content": filtered_content})
                else:
                    storage_messages.append(msg)
                    
            save_chat_history(
                user_id=None,
                thread_id=st.session_state.thread_id,
                messages=storage_messages,
                title=prompt[:50] + "..." if len(prompt) > 50 else prompt
            )

        except Exception as e:
            logger.error(f"Error in chat interaction: {str(e)}")
            st.error("Error communicating with the assistant. Please try again.")
            response_content = [{"type": "text", "content": f"Lo siento, hubo un error: {str(e)}"}]
            st.session_state.messages.append({"role": "assistant", "content": response_content}) 