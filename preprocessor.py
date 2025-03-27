import re
import pandas as pd
import emoji
from urlextract import URLExtract
from datetime import datetime
import html
import gc

# Initialize URL extractor once
extractor = URLExtract()

def parse_chat(data):
    """
    Parse WhatsApp chat data from a .txt file and convert to a DataFrame
    
    Args:
        data (str): Raw chat data read from the WhatsApp export file
    
    Returns:
        df (pd.DataFrame): DataFrame with structured chat data
    """
    # Define patterns for different date formats and message parts
    pattern_12hr = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s[APap][Mm])\s-\s(.*?):\s(.*)'
    pattern_24hr = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s-\s(.*?):\s(.*)'
    
    # Add a new pattern for the [DD.MM.YYYY, HH:MM:SS] format
    pattern_bracket_format = r'\[(\d{1,2}\.\d{1,2}\.\d{4}),\s(\d{1,2}:\d{2}:\d{2})\]\s(.*?):\s(.*)'
    
    # Initialize empty lists to store data
    dates = []
    times = []
    users = []
    messages = []
    
    # Precompile regex patterns for better performance
    regex_12hr = re.compile(pattern_12hr)
    regex_24hr = re.compile(pattern_24hr)
    regex_bracket = re.compile(pattern_bracket_format)
    
    # Split data into lines and process in chunks
    lines = data.split('\n')
    
    # Process in chunks to reduce memory usage
    chunk_size = 5000
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i:i+chunk_size]
        
        for line in chunk_lines:
            # Try to match line with the patterns
            match_12hr = regex_12hr.match(line)
            match_24hr = regex_24hr.match(line)
            match_bracket = regex_bracket.match(line)
            
            if match_12hr:
                dates.append(match_12hr.group(1))
                times.append(match_12hr.group(2))
                users.append(match_12hr.group(3))
                # Store the raw message - will be cleaned when displayed
                messages.append(match_12hr.group(4))
            elif match_24hr:
                dates.append(match_24hr.group(1))
                times.append(match_24hr.group(2))
                users.append(match_24hr.group(3))
                # Store the raw message - will be cleaned when displayed
                messages.append(match_24hr.group(4))
            elif match_bracket:
                dates.append(match_bracket.group(1))
                times.append(match_bracket.group(2))
                users.append(match_bracket.group(3))
                # Store the raw message - will be cleaned when displayed
                messages.append(match_bracket.group(4))
            else:
                # If the line doesn't match a new message pattern, 
                # it's likely a continuation of the previous message
                if len(messages) > 0:
                    messages[-1] += '\n' + line
    
    # Create a DataFrame efficiently
    df = pd.DataFrame({
        'date': dates,
        'time': times,
        'user': users,
        'message': messages
    })
    
    # Use category data type for certain columns to save memory
    if not df.empty:
        df['user'] = df['user'].astype('category')
    
    # Free up memory
    del dates, times, users, messages, lines
    gc.collect()
    
    return df


def preprocess(df):
    """
    Preprocess the DataFrame to extract useful features
    
    Args:
        df (pd.DataFrame): DataFrame with raw chat data
    
    Returns:
        df (pd.DataFrame): Processed DataFrame with additional features
    """
    if df.empty:
        return df
    
    # Convert date string to datetime object more efficiently
    try:
        # First, determine likely date format by examining a sample
        sample_size = min(100, len(df))
        sample_dates = df['date'].iloc[:sample_size]
        
        # Check format based on the first date with a period - fix escape sequence
        if sample_dates.str.contains(r'\.').any():  # DD.MM.YYYY format
            date_format = '%d.%m.%Y'
        elif sample_dates.str.contains('/').any():
            # Get a representative date with slash
            sample_date = sample_dates[sample_dates.str.contains('/')].iloc[0]
            parts = sample_date.split('/')
            if len(parts) == 3:
                # If first component is > 12, it's likely DD/MM format
                if int(parts[0]) > 12:
                    date_format = '%d/%m/%Y' if len(parts[2]) == 4 else '%d/%m/%y'
                else:
                    # Try to infer based on value ranges
                    date_format = '%m/%d/%Y' if len(parts[2]) == 4 else '%m/%d/%y'
            else:
                date_format = None
        else:
            date_format = None
        
        # Apply the determined format
        if date_format:
            df['date'] = pd.to_datetime(df['date'], format=date_format)
        else:
            # Fallback to flexible parsing with dayfirst=True to prioritize DD/MM over MM/DD
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    except Exception as e:
        # Use dayfirst=True as a fallback to prioritize DD/MM/YYYY over MM/DD/YYYY
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    
    # Convert 12-hour format to 24-hour format if needed, using vectorized operations
    def convert_time(time_str):
        if not isinstance(time_str, str):
            return time_str
            
        if 'AM' in time_str or 'PM' in time_str or 'am' in time_str or 'pm' in time_str:
            try:
                return datetime.strptime(time_str, '%I:%M %p').strftime('%H:%M')
            except:
                return time_str
        if ':' in time_str and time_str.count(':') == 2:  # HH:MM:SS format
            try:
                return datetime.strptime(time_str, '%H:%M:%S').strftime('%H:%M')
            except:
                return time_str
        return time_str
    
    # Apply conversion in chunks to avoid memory issues
    chunk_size = 5000
    for i in range(0, len(df), chunk_size):
        chunk_end = min(i + chunk_size, len(df))
        df.loc[i:chunk_end, 'time'] = df.loc[i:chunk_end, 'time'].apply(convert_time)
    
    # Create datetime column
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    
    # Extract additional time features efficiently
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day'] = df['datetime'].dt.day
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month
    df['month_name'] = df['datetime'].dt.month_name()
    df['year'] = df['datetime'].dt.year
    
    # Convert categorical columns to save memory
    df['day_of_week'] = df['day_of_week'].astype('category')
    df['month_name'] = df['month_name'].astype('category')
    
    # Extract message features efficiently
    # Use vectorized operations when possible
    df['message_length'] = df['message'].str.len()
    
    # Handle NaN values safely
    df['word_count'] = df['message'].fillna('').apply(lambda x: len(x.split()))
    
    # Check if message contains media
    media_terms = ['omitted', '<media omitted>']
    df['has_media'] = df['message'].fillna('').str.lower().apply(
        lambda x: 1 if any(term in x for term in media_terms) else 0
    )
    
    # Process URLs directly with vectorized operations instead of chunking
    df['has_url'] = df['message'].fillna('').apply(lambda x: bool(extractor.find_urls(x)))
    df['url_count'] = df['message'].fillna('').apply(lambda x: len(extractor.find_urls(x)))
    
    # Extract emojis in chunks to prevent memory issues
    def extract_emojis(text):
        if not isinstance(text, str):
            return []
        return [c for c in text if c in emoji.EMOJI_DATA]
    
    # Process emojis with vectorized operations
    df['emojis'] = df['message'].fillna('').apply(extract_emojis)
    df['emoji_count'] = df['emojis'].apply(len)
    
    # Clean up memory
    gc.collect()
    
    return df


def analyze_chat(data):
    """
    Main function to process chat data
    
    Args:
        data (str): Raw chat data from .txt file
    
    Returns:
        df (pd.DataFrame): Fully processed DataFrame
    """
    df = parse_chat(data)
    df = preprocess(df)
    return df 