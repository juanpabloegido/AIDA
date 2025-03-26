import streamlit as st
from supabase import create_client
import os
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Supabase client
def init_supabase():
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase = create_client(supabase_url, supabase_key)
    
    # Set session if authenticated
    if "user" in st.session_state and hasattr(st.session_state, "access_token"):
        logger.info("Setting Supabase session")
        logger.info(f"Access token: {st.session_state.access_token[:10]}...")
        supabase.auth.set_session(
            access_token=st.session_state.access_token,
            refresh_token=st.session_state.refresh_token
        )
    else:
        logger.warning("No session tokens found in session_state")
    
    return supabase

def save_chat_history(user_id, thread_id, messages, title=None):
    if not thread_id:
        logger.error("No thread_id provided")
        return
        
    try:
        logger.info(f"Attempting to save chat history for thread_id: {thread_id}")
        supabase = init_supabase()
        
        # Get current user from session
        if not hasattr(st.session_state, "user"):
            logger.error("No user in session_state")
            return
            
        if not hasattr(st.session_state.user, "id"):
            logger.error("User object has no id attribute")
            logger.info(f"User object contents: {st.session_state.user}")
            return
            
        user_id = st.session_state.user.id
        logger.info(f"Current user ID: {user_id}")
        
        # Log session state
        logger.info(f"Session tokens - Access: {st.session_state.get('access_token', 'None')}")
        logger.info(f"User session: {getattr(st.session_state.user, 'session', 'None')}")
            
        # Generate title from first user message if not provided
        if not title and messages:
            for msg in messages:
                if msg["role"] == "user":
                    title = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                    break
            if not title:
                title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        logger.info(f"Generated title: {title}")
        
        # Check if thread exists
        response = supabase.table("chat_history").select("*").eq("thread_id", thread_id).execute()
        logger.info(f"Thread exists check response: {response.data}")
        
        if response.data:
            # Update existing thread
            logger.info(f"Updating existing thread {thread_id} for user {user_id}")
            update_data = {
                "messages": messages,
                "updated_at": datetime.now().isoformat(),
                "title": title
            }
            logger.info(f"Update data: {update_data}")
            
            supabase.table("chat_history").update(update_data).eq("thread_id", thread_id).eq("user_id", user_id).execute()
        else:
            # Create new thread
            logger.info(f"Creating new thread {thread_id} for user {user_id}")
            insert_data = {
                "user_id": user_id,
                "thread_id": thread_id,
                "messages": messages,
                "title": title,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            logger.info(f"Insert data: {insert_data}")
            
            supabase.table("chat_history").insert(insert_data).execute()
            
        logger.info("Chat history saved successfully")
            
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        if hasattr(e, '__dict__'):
            logger.error(f"Error attributes: {e.__dict__}")
        st.error(f"Error saving chat history: {str(e)}")

def load_chat_history(user_id):
    try:
        supabase = init_supabase()
        
        # Get current user from session
        if not hasattr(st.session_state.user, "id"):
            logger.error("No authenticated user found")
            return []
            
        response = supabase.table("chat_history").select("*").eq("user_id", st.session_state.user.id).order("updated_at", desc=True).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error loading chat history: {str(e)}")
        st.error(f"Error loading chat history: {str(e)}")
        return []

def get_chat_thread(thread_id):
    try:
        supabase = init_supabase()
        
        # Get current user from session
        if not hasattr(st.session_state.user, "id"):
            logger.error("No authenticated user found")
            return None
            
        response = supabase.table("chat_history").select("*").eq("thread_id", thread_id).eq("user_id", st.session_state.user.id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error loading chat thread: {str(e)}")
        st.error(f"Error loading chat thread: {str(e)}")
        return None

def delete_chat_thread(thread_id):
    try:
        supabase = init_supabase()
        
        # Get current user from session
        if not hasattr(st.session_state.user, "id"):
            logger.error("No authenticated user found")
            return False
            
        supabase.table("chat_history").delete().eq("thread_id", thread_id).eq("user_id", st.session_state.user.id).execute()
        return True
    except Exception as e:
        logger.error(f"Error deleting chat thread: {str(e)}")
        st.error(f"Error deleting chat thread: {str(e)}")
        return False

def login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("### 🔐 Login")
        
        # Center the login form
        col1, col2, col3 = st.columns([1,2,1])
        
        with col2:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            if st.button("Login", type="primary", use_container_width=True):
                try:
                    supabase = init_supabase()
                    auth_response = supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": password
                    })
                    
                    logger.info("Login successful")
                    logger.info(f"Auth response user: {auth_response.user}")
                    logger.info(f"Auth response session: {auth_response.session}")
                    
                    # Store user data and session in session state
                    st.session_state.user = auth_response.user
                    st.session_state.authenticated = True
                    st.session_state.access_token = auth_response.session.access_token
                    st.session_state.refresh_token = auth_response.session.refresh_token
                    
                    # Set the session in Supabase client
                    supabase.auth.set_session(
                        access_token=auth_response.session.access_token,
                        refresh_token=auth_response.session.refresh_token
                    )
                    
                    logger.info("Session state after login:")
                    logger.info(f"User ID: {st.session_state.user.id}")
                    logger.info(f"Access Token: {st.session_state.access_token[:10]}...")
                    logger.info(f"User object: {st.session_state.user}")
                    
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Login error: {str(e)}")
                    logger.error(f"Error type: {type(e)}")
                    if hasattr(e, '__dict__'):
                        logger.error(f"Error attributes: {e.__dict__}")
                    st.error("Invalid credentials")
            
            # Add register link
            st.markdown("---")
            st.markdown("Don't have an account? [Register here](/register)", unsafe_allow_html=True)
        
        return False
    
    # Refresh session if needed
    try:
        supabase = init_supabase()
        if hasattr(st.session_state, "access_token") and hasattr(st.session_state, "refresh_token"):
            logger.info("Refreshing session")
            logger.info(f"Current user ID: {st.session_state.user.id}")
            logger.info(f"Access Token: {st.session_state.access_token[:10]}...")
            
            supabase.auth.set_session(
                access_token=st.session_state.access_token,
                refresh_token=st.session_state.refresh_token
            )
    except Exception as e:
        logger.error(f"Error refreshing session: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        if hasattr(e, '__dict__'):
            logger.error(f"Error attributes: {e.__dict__}")
        st.session_state.clear()
        st.rerun()
    
    return True

def register():
    st.markdown("### 📝 Register")
    
    # Center the registration form
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Register", type="primary", use_container_width=True):
            if password != confirm_password:
                st.error("Passwords don't match")
                return
                
            try:
                supabase = init_supabase()
                response = supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })
                
                st.success("Registration successful! Please check your email to confirm your account.")
                st.markdown("[Go to login](/)", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")
        
        st.markdown("---")
        st.markdown("Already have an account? [Login here](/)", unsafe_allow_html=True)

def logout():
    if st.sidebar.button("Logout", type="secondary"):
        st.session_state.clear()
        st.rerun() 