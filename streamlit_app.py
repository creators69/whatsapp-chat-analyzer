import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from urlextract import URLExtract
import re
import html
import logging
import emoji
import gc
import os
from functools import lru_cache
import nltk

# Set a specific directory for NLTK data (fix for Streamlit Cloud permission issues)
nltk_data_dir = os.path.expanduser("~/nltk_data")
if not os.path.exists(nltk_data_dir):
    try:
        os.makedirs(nltk_data_dir, exist_ok=True)
    except:
        nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)

nltk.data.path.append(nltk_data_dir)

# Download NLTK data on Streamlit Cloud
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    except:
        st.warning("Unable to download NLTK punkt data. Some features may not work correctly.")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    except:
        st.warning("Unable to download NLTK stopwords data. Some features may not work correctly.")

# Set up logging
logging.basicConfig(level=logging.WARNING)

# Create URL extractor
extractor = URLExtract()

# Check if running in Docker to optimize memory usage
IN_DOCKER = os.environ.get('STREAMLIT_SERVER_HEADLESS', '') == 'true'

# Memory-saving configuration for matplotlib
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 80

# Helper function to clean message text for display
def clean_message_for_display(message_text):
    """
    Clean message text for proper display by removing HTML tags and escaping special characters
    
    Args:
        message_text (str): Raw message text
        
    Returns:
        str: Cleaned message text safe for HTML display
    """
    if not message_text:
        return ""
    
    try:
        logging.debug(f"Original message: {message_text}")
        # Escape HTML special characters
        message_text = html.escape(message_text)
        logging.debug(f"Escaped message: {message_text}")
        
        # Extract URLs and make them clickable
        urls = extractor.find_urls(message_text)
        for url in urls:
            message_text = message_text.replace(html.escape(url), f'<a href="{url}" target="_blank" style="color: #2196F3; text-decoration: underline;">{url}</a>')
        logging.debug(f"Final cleaned message: {message_text}")
        
        return message_text
    except Exception as e:
        # If anything fails, return a simple escaped version of the text
        logging.error(f"Error cleaning message: {str(e)}")
        return html.escape(str(message_text))

# Add this helper function after the existing import statements
def clean_message_content(message):
    """
    Clean the message content for display:
    - Handle HTML/CSS code snippets
    - Escape HTML characters
    - Preserve emojis
    """
    if not message or not isinstance(message, str):
        return ""
    
    # Use more robust regex pattern matching for HTML detection
    html_tag_pattern = re.compile(r'</?(?:div|span|p|a|b|i|u|strong|em|table|tr|td|th|ul|ol|li|br|hr|img|style)[^>]*>|style\s*=\s*["\'][^"\']*["\']')
    css_property_pattern = re.compile(r'(?:font-size|color|margin|padding|overflow-wrap|white-space|word-wrap|text-align|line-height|border-radius|background-color)\s*:')
    angle_bracket_pattern = re.compile(r'<[^>]+>')
    
    # Check if message appears to be HTML/CSS code using multiple criteria
    is_html_code = bool(html_tag_pattern.search(message) or css_property_pattern.search(message))
    
    # Additional checks for HTML-like content
    if not is_html_code:
        # Check for angle brackets that might indicate HTML tags
        if angle_bracket_pattern.search(message):
            is_html_code = True
        
        # Check for suspicious patterns like multiple style attributes
        elif 'style=' in message.lower() or 'font-size' in message.lower() or 'color:' in message.lower():
            is_html_code = True
        
        # Check angle bracket pairs - if there are multiple, it's likely HTML
        elif message.count('<') > 1 and message.count('>') > 1:
            is_html_code = True
    
    if is_html_code:
        # Just show a placeholder instead of the HTML code
        return "[Message with formatting]"
    
    # For messages with just emojis, return as is without escaping
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    
    # If message is only emoji or whitespace+emoji, return as is
    if emoji_pattern.sub('', message.strip()) == '':
        return message
        
    # Regular message - escape any HTML characters safely
    return html.escape(message)

# Import our modules
from preprocessor import analyze_chat
from analytics import ChatAnalyzer

# Set page config
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: 600;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .subtitle-text {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin-bottom: 1.5rem;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #1e88e5;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #eee;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Page header
st.markdown("<div class='title-text'>WhatsApp Chat Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Upload your WhatsApp chat export to get detailed insights and visualizations</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/WhatsApp.svg/512px-WhatsApp.svg.png", width=100)
st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat export file", type=["txt"])

if uploaded_file is not None:
    # Read file as text
    bytes_data = uploaded_file.getvalue()
    try:
        # Try to decode with utf-8
        data = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        # If utf-8 fails, try with ISO-8859-1
        data = bytes_data.decode("ISO-8859-1")
    
    # Process the chat data
    df = analyze_chat(data)
    
    # Check if parsing was successful
    if df.empty:
        st.error("No messages found in the file. Please check if this is a valid WhatsApp chat export.")
        st.info("Make sure the file format matches WhatsApp's export format: 'MM/DD/YY, HH:MM - Sender: Message'")
        st.stop()
    
    # Create the analyzer
    analyzer = ChatAnalyzer(df)
    
    # Get unique users for filtering
    users = df['user'].unique().tolist()
    users.insert(0, "All Users")
    
    # Sidebar - User selection
    selected_user = st.sidebar.selectbox("Select User", users)
    
    # Filter data based on user selection
    if selected_user != "All Users":
        df_filtered = df[df['user'] == selected_user]
        analyzer_filtered = ChatAnalyzer(df_filtered)
    else:
        df_filtered = df
        analyzer_filtered = analyzer
    
    # Get statistics
    stats = analyzer_filtered.get_basic_stats()
    
    # Main content area
    
    # Basic statistics dashboard layout
    st.markdown("<div class='section-header'>Chat Overview</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-number'>{stats['total_messages']}</div>", unsafe_allow_html=True)
        st.markdown("<div class='stat-label'>Total Messages</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-number'>{stats['total_words']}</div>", unsafe_allow_html=True)
        st.markdown("<div class='stat-label'>Total Words</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-number'>{stats['media_shared']}</div>", unsafe_allow_html=True)
        st.markdown("<div class='stat-label'>Media Shared</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-number'>{stats['links_shared']}</div>", unsafe_allow_html=True)
        st.markdown("<div class='stat-label'>Links Shared</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-number'>{stats['chat_duration_days']}</div>", unsafe_allow_html=True)
        st.markdown("<div class='stat-label'>Chat Duration (Days)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-number'>{stats['total_emojis']}</div>", unsafe_allow_html=True)
        st.markdown("<div class='stat-label'>Total Emojis</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat timeline and trends
    st.markdown("<div class='section-header'>Message Trends</div>", unsafe_allow_html=True)
    
    message_trend_fig = analyzer_filtered.create_message_trend_plot()
    if message_trend_fig:
        st.plotly_chart(message_trend_fig, use_container_width=True)
    
    # Activity patterns
    st.markdown("<div class='section-header'>Activity Patterns</div>", unsafe_allow_html=True)
    
    try:
        # Only calculate when the tab is active 
        activity_data = analyzer_filtered.get_activity_by_time()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly activity
            hourly = activity_data['hourly_activity']
            if not hourly.empty:
                fig = px.bar(
                    hourly, 
                    x='hour', 
                    y='message_count',
                    title='Activity by Hour of Day',
                    labels={'hour': 'Hour of Day', 'message_count': 'Number of Messages'},
                    color_discrete_sequence=['#1e88e5']
                )
                fig.update_layout(
                    xaxis=dict(tickmode='linear', dtick=1),
                    height=350, 
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to show hourly activity.")
        
        with col2:
            # Day of week activity
            day_of_week = activity_data['day_of_week_activity']
            if not day_of_week.empty:
                fig = px.bar(
                    day_of_week, 
                    x='day_of_week', 
                    y='message_count',
                    title='Activity by Day of Week',
                    labels={'day_of_week': 'Day of Week', 'message_count': 'Number of Messages'},
                    color_discrete_sequence=['#43a047']
                )
                fig.update_layout(
                    height=350, 
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to show daily activity.")
        
        # Only generate heatmap if user requests it (to save memory)
        if st.checkbox("Show Activity Heatmap"):
            with st.spinner("Generating activity heatmap..."):
                heatmap_b64 = analyzer_filtered.create_user_activity_heatmap(
                    user=selected_user if selected_user != "All Users" else None
                )
                
                if heatmap_b64:
                    st.markdown("<div class='section-header'>Activity Heatmap</div>", unsafe_allow_html=True)
                    st.markdown(f"<img src='data:image/png;base64,{heatmap_b64}' style='width:100%;'>", unsafe_allow_html=True)
                else:
                    st.info("No activity data available to generate heatmap.")
    
    except Exception as e:
        st.error(f"Error generating activity analysis: {str(e)}")
        st.info("Try selecting a different user or date range with more data.")
    
    # Clean memory after generating charts
    gc.collect()
    
    # User analysis
    if selected_user == "All Users" and len(users) > 1:
        st.markdown("<div class='section-header'>User Analysis</div>", unsafe_allow_html=True)
        
        # Top users
        user_stats = analyzer.get_active_users(top_n=10)
        if not user_stats.empty:
            fig = px.bar(
                user_stats, 
                x='User', 
                y='Messages',
                title='Most Active Users',
                color='Messages',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # User comparison for top 5
            user_comparison = analyzer.create_user_comparison_plot(top_n=5)
            if user_comparison:
                st.plotly_chart(user_comparison, use_container_width=True)
    
    # NEW CHAT REPLAY FEATURE
    st.markdown("<div class='section-header'>Chat Replay</div>", unsafe_allow_html=True)
    
    # Add custom CSS for modal and message styling
    st.markdown("""
    <style>
    /* Modal Background */
    .chat-modal-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: 1000;
        backdrop-filter: blur(5px);
    }
    
    .chat-modal {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: #f7f9fa;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        width: 90%;
        max-width: 900px;
        max-height: 85vh;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        z-index: 1001;
    }
    
    .modal-header {
        background-color: #128C7E;
        color: white;
        padding: 15px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-top-left-radius: 16px;
        border-top-right-radius: 16px;
    }
    
    .modal-header-title {
        display: flex;
        align-items: center;
    }
    
    .modal-header-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #E1F3F5;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 15px;
        font-weight: bold;
        font-size: 18px;
        color: #128C7E;
    }
    
    .modal-title {
        font-size: 18px;
        font-weight: 600;
    }
    
    .modal-subtitle {
        font-size: 12px;
        opacity: 0.8;
    }
    
    .modal-close {
        cursor: pointer;
        font-size: 22px;
    }
    
    .modal-content {
        padding: 20px;
        overflow-y: auto;
        flex: 1;
        background-color: #e5ddd5;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M54.627 0l.83.828-1.415 1.415L51.8 0h2.827zM5.373 0l-.83.828L5.96 2.243 8.2 0H5.374zM48.97 0l3.657 3.657-1.414 1.414L46.143 0h2.828zM11.03 0L7.372 3.657 8.787 5.07 13.857 0H11.03zm32.284 0L49.8 6.485 48.384 7.9l-7.9-7.9h2.83zM16.686 0L10.2 6.485 11.616 7.9l7.9-7.9h-2.83zm20.97 0l9.315 9.314-1.414 1.414L34.828 0h2.83zM22.344 0L13.03 9.314l1.414 1.414L25.172 0h-2.83zM32 0l12.142 12.142-1.414 1.414L30 .828 17.272 13.556l-1.414-1.414L28 0h4zM.284 0l28 28-1.414 1.414L0 2.544V0h.284zM0 5.373l25.456 25.455-1.414 1.415L0 8.2V5.374zm0 5.656l22.627 22.627-1.414 1.414L0 13.86v-2.83zm0 5.656l19.8 19.8-1.415 1.413L0 19.514v-2.83zm0 5.657l16.97 16.97-1.414 1.415L0 25.172v-2.83zM0 28l14.142 14.142-1.414 1.414L0 30.828V28zm0 5.657L11.314 44.97 9.9 46.386l-9.9-9.9v-2.828zm0 5.657L8.485 47.8 7.07 49.212 0 42.143v-2.83zm0 5.657l5.657 5.657-1.414 1.415L0 47.8v-2.83zm0 5.657l2.828 2.83-1.414 1.413L0 53.456v-2.83zM54.627 60L30 35.373 5.373 60H8.2L30 38.2 51.8 60h2.827zm-5.656 0L30 41.03 11.03 60h2.828L30 43.858 46.142 60h2.83zm-5.656 0L30 46.686 16.686 60h2.83L30 49.515 40.485 60h2.83zm-5.657 0L30 52.343 22.344 60h2.83L30 55.172 34.828 60h2.83zM32 60l-2-2-2 2h4zM59.716 0l-28 28 1.414 1.414L60 2.544V0h-.284zM60 5.373L34.544 30.828l1.414 1.415L60 8.2V5.374zm0 5.656L37.373 33.656l1.414 1.414L60 13.86v-2.83zm0 5.656l-19.8 19.8 1.415 1.413L60 19.514v-2.83zm0 5.657l-16.97 16.97 1.414 1.415L60 25.172v-2.83zM60 28L45.858 42.142l1.414 1.414L60 30.828V28zm0 5.657L48.686 44.97l1.415 1.415 9.9-9.9v-2.828zm0 5.657L51.515 47.8l1.414 1.414L60 42.143v-2.83zm0 5.657l-5.657 5.657 1.414 1.415L60 47.8v-2.83zm0 5.657l-2.828 2.83 1.414 1.413L60 53.456v-2.83z' fill='%23dcf8c6' fill-opacity='0.12' fill-rule='evenodd'/%3E%3C/svg%3E");
    }
    
    .message-time-header {
        text-align: center;
        margin: 15px 0;
        position: relative;
    }
    
    .message-time-header span {
        background-color: rgba(225, 245, 254, 0.7);
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        color: #075E54;
        box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
    }
    
    .message-row {
        display: flex;
        margin-bottom: 10px;
        width: 100%;
        align-items: flex-start;
    }
    
    .message-row.right {
        justify-content: flex-end;
    }
    
    .message-bubble {
        padding: 10px 14px;
        border-radius: 12px;
        max-width: 75%;
        position: relative;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        animation: fadeIn 0.3s ease;
    }
    
    .left-bubble {
        background-color: white;
        border-top-left-radius: 4px;
        margin-right: auto;
    }
    
    .left-bubble::before {
        content: "";
        position: absolute;
        top: 0;
        left: -8px;
        width: 8px;
        height: 13px;
        background-color: white;
        border-top-right-radius: 13px;
        clip-path: polygon(0 0, 100% 0, 100% 100%);
    }
    
    .right-bubble {
        background-color: #DCF8C6;
        border-top-right-radius: 4px;
        margin-left: auto;
    }
    
    .right-bubble::after {
        content: "";
        position: absolute;
        top: 0;
        right: -8px;
        width: 8px;
        height: 13px;
        background-color: #DCF8C6;
        border-top-left-radius: 13px;
        clip-path: polygon(0 0, 100% 0, 0 100%);
    }
    
    .message-user {
        font-weight: 600;
        color: #5E35B1;
        margin-bottom: 4px;
        font-size: 13px;
    }
    
    .message-content {
        font-size: 14px;
        line-height: 1.4;
        color: #303030;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
    }
    
    .message-time {
        font-size: 10px;
        color: #8D8D8D;
        text-align: right;
        margin-top: 3px;
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }
    
    .read-receipt {
        color: #4FC3F7;
        margin-left: 3px;
    }
    
    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: #E1F3F5;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
        color: white;
        margin-right: 8px;
        flex-shrink: 0;
    }
    
    .chat-instruction {
        background-color: #EFF6F9;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #128C7E;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .calendar-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    .custom-button {
        background-color: #128C7E;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-top: 10px;
    }
    
    .custom-button:hover {
        background-color: #0E7669;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }
    
    .custom-button:active {
        transform: translateY(1px);
    }
    
    .custom-button:disabled {
        background-color: #B0BEC5;
        cursor: not-allowed;
    }
    
    .button-icon {
        margin-right: 8px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Modal toggle mechanism */
    .modal-show {
        display: block;
    }
    
    /* Media styles */
    .media-placeholder {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 8px;
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    
    .media-icon {
        font-size: 20px;
        margin-right: 10px;
        color: #128C7E;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .chat-modal {
            width: 95%;
            max-height: 90vh;
        }
        
        .message-bubble {
            max-width: 85%;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add JavaScript for modal functionality
    st.markdown("""
    <script>
    function openChatModal() {
        document.getElementById("chatModal").classList.add("modal-show");
    }
    
    function closeChatModal() {
        document.getElementById("chatModal").classList.remove("modal-show");
    }
    
    // Close modal if clicking outside of it
    window.onclick = function(event) {
        var modal = document.getElementById("chatModal");
        if (event.target == modal) {
            closeChatModal();
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='chat-instruction'>", unsafe_allow_html=True)
    st.markdown("### View WhatsApp Chat Conversations")
    st.markdown("Select a date below to view the conversation from that day.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Calendar view for date selection 
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='calendar-container'>", unsafe_allow_html=True)
        
        # Get the min and max dates from the dataset
        min_date = df_filtered['date'].min().date()
        max_date = df_filtered['date'].max().date()
        
        # Date selector
        selected_date = st.date_input(
            "Select date to view conversation",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        # Get conversation for selected date
        day_conversation = analyzer_filtered.get_conversation_by_date(selected_date)
        
        message_count = len(day_conversation)
        
        # If no messages, show info
        if day_conversation.empty:
            st.info(f"No messages found for {selected_date}")
        else:
            # View chat button
            if st.button(f"View Conversation ({message_count} messages)", key="view_chat_button"):
                st.session_state.show_chat = True
                st.session_state.selected_date = selected_date
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Show message statistics for the selected day
        if not day_conversation.empty:
            st.markdown("<div style='margin-top: 23px;'>", unsafe_allow_html=True)
            st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-number'>{message_count}</div>", unsafe_allow_html=True)
            st.markdown("<div class='stat-label'>Messages on this date</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show unique users on that day
            unique_users = day_conversation['user'].nunique()
            st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-number'>{unique_users}</div>", unsafe_allow_html=True)
            st.markdown("<div class='stat-label'>Active Users</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Display conversation if show_chat is True
    if st.session_state.get('show_chat', False):
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        date_to_show = st.session_state.get('selected_date', selected_date)
        conversation_to_show = analyzer_filtered.get_conversation_by_date(date_to_show)
        
        if conversation_to_show.empty:
            st.warning(f"No messages found for {date_to_show}")
        else:
            # Create chat header with native components
            st.subheader("WhatsApp Chat")
            st.caption(f"{date_to_show.strftime('%A, %B %d, %Y')} • {len(conversation_to_show)} messages")
            
            # Create a container for the chat
            chat_area = st.container()
            
            # Apply minimal CSS
            st.markdown("""
            <style>
            .streamlit-container {
                max-width: 1000px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Determine primary user (most active user in the conversation)
            if selected_user != "All Users":
                primary_user = selected_user
            else:
                primary_user = conversation_to_show['user'].value_counts().idxmax()
            
            # Create user color mapping
            users = conversation_to_show['user'].unique()
            user_colors = {}
            color_options = ['#128C7E', '#075E54', '#25D366', '#34B7F1', '#5E35B1', '#673AB7', '#3949AB', '#1E88E5', '#039BE5', '#00ACC1']
            
            for idx, user in enumerate(users):
                user_colors[user] = color_options[idx % len(color_options)]
            
            # Group messages by hour
            last_time_hour = None
            last_user = None
            
            # Process and display messages within the chat container
            with chat_area:
                for idx, msg in conversation_to_show.iterrows():
                    # Check if we need to add time header
                    current_hour = msg['datetime'].hour
                    if last_time_hour is None or current_hour != last_time_hour:
                        timestamp = msg['datetime'].strftime('%I:%M %p')
                        st.caption(f"───────── {timestamp} ─────────")
                        last_time_hour = current_hour
                    
                    # Determine message alignment
                    alignment = "right" if msg['user'] == primary_user else "left"
                    
                    # Show username if different from previous sender
                    show_user = last_user != msg['user']
                    last_user = msg['user']
                    
                    # Get raw message without any HTML
                    if msg['message'] and isinstance(msg['message'], str):
                        # Strip all HTML tags
                        clean_message = re.sub(r'<.*?>', '', msg['message'])
                        # Remove CSS properties
                        clean_message = re.sub(r'style\s*=\s*["\'][^"\']*["\']', '', clean_message)
                        # If we removed everything, it was probably HTML
                        if clean_message.strip() == "" and ("<" in msg['message'] or "style=" in msg['message']):
                            clean_message = "[Message with formatting]"
                    else:
                        clean_message = ""
                    
                    # Format time
                    time_str = msg['datetime'].strftime('%H:%M')
                    
                    # Use columns for message layout - simpler approach
                    cols = st.columns([10])
                    
                    if alignment == "left":
                        # Left message with simple markdown
                        with cols[0]:
                            user_name = msg['user']
                            user_color = user_colors.get(user_name, '#128C7E')
                            
                            if show_user:
                                st.markdown(f"**{user_name}**")
                            
                            if msg['has_media']:
                                st.info("🖼️ Media attachment", icon="📱")
                            else:
                                st.markdown(f"```\n{clean_message}\n```")
                            
                            st.caption(f"{time_str}")
                    else:
                        # Right message with simple markdown
                        with cols[0]:
                            st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
                            
                            if show_user:
                                st.markdown(f"**{msg['user']}**")
                            
                            if msg['has_media']:
                                st.info("🖼️ Media attachment", icon="📱")
                            else:
                                st.success(clean_message)
                            
                            st.caption(f"{time_str} ✓✓")
                            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add close button
            if st.button("Close Conversation", key="close_chat_button"):
                st.session_state.show_chat = False
                st.experimental_rerun()

    # Word analysis
    st.markdown("<div class='section-header'>Word Analysis</div>", unsafe_allow_html=True)
    
    try:
        # Only generate word analysis when requested
        if st.checkbox("Generate Word Analysis", value=True):
            with st.spinner("Analyzing word patterns..."):
                word_analysis = analyzer_filtered.get_word_analysis(top_n=20)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Most common words
                    word_freq = word_analysis['word_freq']
                    if not word_freq.empty:
                        fig = px.bar(
                            word_freq.head(10), 
                            x='frequency', 
                            y='word',
                            title='Most Common Words',
                            orientation='h',
                            color='frequency',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No word frequency data available. Try including more messages.")
                
                with col2:
                    # Word cloud
                    wordcloud_b64 = word_analysis['wordcloud_b64']
                    if wordcloud_b64:
                        st.markdown("<h4>Word Cloud</h4>", unsafe_allow_html=True)
                        st.markdown(f"<img src='data:image/png;base64,{wordcloud_b64}' style='width:100%;'>", unsafe_allow_html=True)
                    else:
                        st.info("Not enough data to generate a word cloud.")
        
        # Show emoji analysis - but only generate when requested
        if stats['total_emojis'] > 0 and st.checkbox("Show Emoji Analysis", value=True if stats['total_emojis'] > 0 else False):
            with st.spinner("Analyzing emoji usage..."):
                emoji_freq = analyzer_filtered.get_emoji_analysis(top_n=10)
                
                if not emoji_freq.empty:
                    st.markdown("<div class='section-header'>Emoji Analysis</div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.dataframe(emoji_freq, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(
                            emoji_freq, 
                            values='frequency', 
                            names='emoji',
                            title='Emoji Distribution',
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No emoji data available for analysis.")
        elif stats['total_emojis'] == 0:
            st.info("No emojis found in the selected messages.")
    
    except Exception as e:
        st.error(f"Error in word analysis: {str(e)}")
        st.info("This could be due to insufficient data or unsupported text formats.")
    
    # Free memory
    gc.collect()

    # Media analysis
    if stats['media_shared'] > 0:
        st.markdown("<div class='section-header'>Media Sharing Analysis</div>", unsafe_allow_html=True)
        
        media_analysis = analyzer_filtered.get_media_analysis()
        
        if selected_user == "All Users" and not media_analysis['media_by_user'].empty:
            # Media by user
            fig = px.bar(
                media_analysis['media_by_user'].head(10), 
                x='user', 
                y='media_count',
                title='Media Shared by User',
                color='media_count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Media over time
        if not media_analysis['media_over_time'].empty:
            fig = px.line(
                media_analysis['media_over_time'], 
                x='year_month', 
                y='has_media',
                title='Media Shared Over Time',
                markers=True,
                labels={'year_month': 'Month', 'has_media': 'Media Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    # Show instructions when no file is uploaded
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("### How to export your WhatsApp chat", unsafe_allow_html=True)
    st.markdown("""
    1. Open the WhatsApp conversation you want to analyze
    2. Tap the three dots in the top right corner
    3. Select 'More' > 'Export chat'
    4. Choose 'Without Media' for faster upload
    5. Save the exported .txt file
    6. Upload the file using the sidebar on the left
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show sample dashboard
    st.markdown("<div class='section-header'>Sample Dashboard Preview</div>", unsafe_allow_html=True)
    st.image("https://miro.medium.com/max/1400/1*Ry8EMP-KbHmaBX_LGKQUEg.png", caption="Sample Analysis Dashboard")
    
    # Feature highlights
    st.markdown("<div class='section-header'>Features</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown("##### 📊 Chat Statistics")
        st.markdown("Get insights on message counts, word usage, media sharing, and more.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown("##### 👥 User Analysis")
        st.markdown("Identify most active participants and compare their chatting patterns.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown("##### ⏱️ Time Patterns")
        st.markdown("Discover when conversations are most active with hourly, daily, and monthly breakdowns.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown("##### 🔤 Word Analysis")
        st.markdown("Explore the most common words and generate beautiful word clouds.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown("##### 😀 Emoji Usage")
        st.markdown("See which emojis are used most frequently in your conversations.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.markdown("##### 📈 Interactive Visualizations")
        st.markdown("Engage with dynamic charts and graphs to explore your chat data.")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem 0;'>"
    "WhatsApp Chat Analyzer • A powerful tool for analyzing your conversations"
    "</div>", 
    unsafe_allow_html=True
) 