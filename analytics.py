import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import io
import base64
from PIL import Image
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gc  # Garbage collection

# Import custom stop words
from stop_words import get_stopwords

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ChatAnalyzer:
    def __init__(self, df):
        """
        Initialize the analyzer with the preprocessed DataFrame
        
        Args:
            df (pd.DataFrame): Preprocessed chat DataFrame
        """
        # Use pandas optimizations for memory efficiency
        self.df = df.copy()
        
        # Convert object columns to categories where appropriate
        for col in ['user', 'day_of_week', 'month_name']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
        
        # Combine NLTK stopwords with our custom stopwords
        nltk_stopwords = set(stopwords.words('english'))
        custom_stopwords = get_stopwords()
        self.stop_words = nltk_stopwords.union(custom_stopwords)
        
    def get_basic_stats(self):
        """
        Get basic statistics about the chat
        """
        if self.df.empty:
            return {
                'total_messages': 0,
                'total_words': 0,
                'media_shared': 0,
                'links_shared': 0,
                'total_emojis': 0
            }
            
        stats = {
            'total_messages': len(self.df),
            'total_words': int(self.df['word_count'].sum()),
            'media_shared': int(self.df['has_media'].sum()),
            'links_shared': int(self.df['url_count'].sum()),
            'total_emojis': int(self.df['emoji_count'].sum()),
            'chat_duration_days': (self.df['date'].max() - self.df['date'].min()).days + 1,
            'first_message_date': self.df['date'].min().strftime('%Y-%m-%d'),
            'last_message_date': self.df['date'].max().strftime('%Y-%m-%d'),
            'avg_messages_per_day': len(self.df) / ((self.df['date'].max() - self.df['date'].min()).days + 1)
        }
        return stats
        
    def get_active_users(self, top_n=10):
        """
        Get the most active users in the chat
        
        Args:
            top_n (int): Number of top users to return
            
        Returns:
            pd.DataFrame: DataFrame with user activity stats
        """
        if self.df.empty:
            return pd.DataFrame()
            
        # Use optimized groupby
        user_stats = self.df.groupby('user').agg({
            'message': 'count',
            'word_count': 'sum',
            'message_length': 'mean',
            'has_media': 'sum',
            'url_count': 'sum',
            'emoji_count': 'sum'
        }).reset_index()
        
        user_stats.columns = ['User', 'Messages', 'Words', 'Avg Message Length', 
                              'Media Shared', 'Links Shared', 'Emojis Used']
        
        result = user_stats.sort_values(by='Messages', ascending=False).head(top_n)
        
        # Clear memory
        del user_stats
        gc.collect()
        
        return result
        
    def get_activity_by_time(self):
        """
        Get activity patterns by hour, day, and month
        
        Returns:
            dict: Dictionary with various time-based activity data
        """
        if self.df.empty:
            return {
                'hourly_activity': pd.DataFrame(),
                'daily_activity': pd.DataFrame(),
                'monthly_activity': pd.DataFrame(),
                'day_of_week_activity': pd.DataFrame()
            }
            
        # Hourly activity - use more efficient operations
        hourly = self.df.groupby('hour', as_index=False)['message'].count()
        hourly.columns = ['hour', 'message_count']
        
        # Daily activity - only calculate if needed for visualization
        # Use DatetimeIndex for more efficient grouping
        daily = pd.DataFrame({'message_count': self.df.groupby(self.df['date'].dt.date).size()})
        daily.reset_index(inplace=True)
        daily.columns = ['date', 'message_count']
        
        # Monthly activity with efficient string operations - fixed to avoid column naming issues
        # Create year_month field first
        self.df['year_month_str'] = self.df['datetime'].dt.strftime('%Y-%m')
        monthly = self.df.groupby('year_month_str').size().reset_index(name='message_count')
        monthly.columns = ['year_month', 'message_count']
        
        # Day of week activity
        day_of_week = self.df.groupby('day_of_week', as_index=False)['message'].count()
        day_of_week.columns = ['day_of_week', 'message_count']
        # Ensure correct order of days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_of_week['day_of_week'] = pd.Categorical(day_of_week['day_of_week'], categories=day_order, ordered=True)
        day_of_week = day_of_week.sort_values('day_of_week')
        
        return {
            'hourly_activity': hourly,
            'daily_activity': daily,
            'monthly_activity': monthly,
            'day_of_week_activity': day_of_week
        }
        
    def get_word_analysis(self, top_n=20, min_length=3):
        """
        Analyze the most common words in the chat
        
        Args:
            top_n (int): Number of top words to return
            min_length (int): Minimum word length to consider
            
        Returns:
            dict: Dictionary with word frequencies and analysis
        """
        if self.df.empty:
            return {
                'word_freq': pd.DataFrame(),
                'wordcloud_b64': None
            }
            
        # Process messages in chunks to save memory
        chunk_size = 1000
        words_counter = Counter()
        
        # Use list comprehension instead of multiple operations
        for i in range(0, len(self.df), chunk_size):
            chunk = self.df['message'].iloc[i:i+chunk_size]
            
            # Clean and tokenize all messages in one operation
            text = ' '.join(chunk.fillna(''))
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            
            # Simple tokenization using split() instead of nltk.word_tokenize
            words = [word for word in text.lower().split() 
                    if word not in self.stop_words and len(word) >= min_length]
            
            # Update counter
            words_counter.update(words)
            
            # Force garbage collection after each chunk
            del chunk, text, words
            gc.collect()
        
        # Get word frequency
        word_freq = words_counter.most_common(top_n)
        word_freq_df = pd.DataFrame(word_freq, columns=['word', 'frequency'])
        
        # Generate wordcloud only if there's enough data
        if words_counter:
            # Optimize wordcloud generation with max_words parameter
            wordcloud = WordCloud(
                width=600,  # Reduce dimensions
                height=300,
                background_color='white',
                max_words=100,  # Limit number of words
                max_font_size=80
            ).generate_from_frequencies(dict(word_freq))
            
            # Compress image before encoding to reduce size
            img = io.BytesIO()
            wordcloud.to_image().save(img, format='PNG', optimize=True, quality=85)
            img.seek(0)
            wordcloud_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
            
            # Clean up
            del wordcloud, img
        else:
            wordcloud_b64 = None
        
        # Clean up
        del words_counter
        gc.collect()
        
        return {
            'word_freq': word_freq_df,
            'wordcloud_b64': wordcloud_b64
        }
        
    def get_emoji_analysis(self, top_n=20):
        """
        Analyze emoji usage in the chat
        
        Args:
            top_n (int): Number of top emojis to return
            
        Returns:
            pd.DataFrame: DataFrame with emoji frequencies
        """
        if self.df.empty or self.df['emoji_count'].sum() == 0:
            return pd.DataFrame()
            
        # Extract all emojis
        all_emojis = []
        for emoji_list in self.df['emojis']:
            if emoji_list:  # Check if list is not empty
                all_emojis.extend(emoji_list)
            
        if not all_emojis:
            return pd.DataFrame()
            
        # Count emoji frequencies
        emoji_freq = Counter(all_emojis).most_common(top_n)
        emoji_freq_df = pd.DataFrame(emoji_freq, columns=['emoji', 'frequency'])
        
        return emoji_freq_df
        
    def get_media_analysis(self):
        """
        Analyze media sharing patterns
        
        Returns:
            dict: Dictionary with media sharing statistics
        """
        if self.df.empty:
            return {
                'total_media': 0,
                'media_by_user': pd.DataFrame(),
                'media_over_time': pd.DataFrame()
            }
            
        total_media = int(self.df['has_media'].sum())
        
        # Media shared by user
        media_by_user = self.df.groupby('user')['has_media'].sum().reset_index()
        media_by_user.columns = ['user', 'media_count']
        media_by_user = media_by_user.sort_values('media_count', ascending=False)
        
        # Media shared over time - use the same approach as monthly activity
        # Using consistent year_month_str column from datetime
        if 'year_month_str' not in self.df.columns:
            self.df['year_month_str'] = self.df['datetime'].dt.strftime('%Y-%m')
            
        media_over_time = self.df.groupby('year_month_str')['has_media'].sum().reset_index()
        media_over_time.columns = ['year_month', 'has_media']
        
        return {
            'total_media': total_media,
            'media_by_user': media_by_user,
            'media_over_time': media_over_time
        }
        
    def create_user_activity_heatmap(self, user=None):
        """
        Create a heatmap of user activity by hour and day
        
        Args:
            user (str): Filter for a specific user, or None for all users
            
        Returns:
            str: Base64 encoded PNG image of the heatmap
        """
        # Filter data if user is provided
        if user and user != "All Users":
            data = self.df[self.df['user'] == user]
        else:
            data = self.df
        
        # Early return if data is empty
        if data.empty:
            return None
        
        # Only use needed columns
        data = data[['hour', 'day_of_week']]
        
        # Create pivot table
        pivot_table = pd.pivot_table(
            data=data,
            index='day_of_week',
            columns='hour',
            values='hour',
            aggfunc='count',
            fill_value=0
        )
        
        # Check if pivot table is empty or has no data
        if pivot_table.empty or pivot_table.values.sum() == 0:
            # Create a simple "No Data" image
            plt.figure(figsize=(8, 4), dpi=80)
            plt.text(0.5, 0.5, "No activity data available for the selected user", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12)
            plt.axis('off')
            
            # Convert to base64
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=80)
            plt.close()
            img.seek(0)
            
            return base64.b64encode(img.getvalue()).decode('utf-8')
        
        # Create a new figure with reduced size
        plt.figure(figsize=(8, 4), dpi=80)
        
        # Order days correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = pivot_table.reindex(day_order)
        
        # Create heatmap with optimized settings
        ax = sns.heatmap(
            pivot_table, 
            cmap='YlGnBu',
            linewidths=0.1,
            linecolor='white',
            annot=False,
            cbar=True
        )
        
        # Set title and labels with reduced font sizes
        title = f"Activity Heatmap: {user}" if user and user != "All Users" else "Activity Heatmap: All Users"
        plt.title(title, fontsize=10)
        plt.xlabel('Hour of Day', fontsize=8)
        plt.ylabel('Day of Week', fontsize=8)
        
        # Set tick labels font size
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        
        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=80, optimize=True)
        plt.close()
        img.seek(0)
        
        heatmap_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
        
        # Clear memory
        del data, pivot_table, img
        gc.collect()
        
        return heatmap_b64
        
    def create_message_trend_plot(self):
        """
        Create an interactive line plot of message trends over time
        
        Returns:
            Plotly figure
        """
        if self.df.empty:
            return None
            
        # Resample by day
        daily_messages = self.df.set_index('datetime').resample('D')['message'].count().reset_index()
        daily_messages.columns = ['date', 'message_count']
        
        # Create plot
        fig = px.line(
            daily_messages, 
            x='date', 
            y='message_count',
            title='Message Trends Over Time',
            labels={'date': 'Date', 'message_count': 'Number of Messages'}
        )
        
        # Add moving average
        daily_messages['moving_avg_7d'] = daily_messages['message_count'].rolling(window=7).mean()
        fig.add_scatter(
            x=daily_messages['date'], 
            y=daily_messages['moving_avg_7d'],
            mode='lines',
            name='7-Day Moving Average',
            line=dict(color='red', width=2)
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Number of Messages',
            legend_title='Legend',
            hovermode='x unified'
        )
        
        return fig
        
    def create_user_comparison_plot(self, top_n=5):
        """
        Create a plot comparing top users by different metrics
        
        Args:
            top_n (int): Number of top users to include
            
        Returns:
            Plotly figure
        """
        if self.df.empty:
            return None
            
        # Get top users
        top_users = self.df['user'].value_counts().nlargest(top_n).index.tolist()
        
        # Filter data for top users
        df_top = self.df[self.df['user'].isin(top_users)]
        
        # Prepare data for subplots
        user_messages = df_top.groupby('user').size().reset_index(name='count')
        user_messages = user_messages.sort_values('count', ascending=False)
        
        user_words = df_top.groupby('user')['word_count'].sum().reset_index()
        user_words = user_words.sort_values('word_count', ascending=False)
        
        user_media = df_top.groupby('user')['has_media'].sum().reset_index()
        user_media = user_media.sort_values('has_media', ascending=False)
        
        user_emojis = df_top.groupby('user')['emoji_count'].sum().reset_index()
        user_emojis = user_emojis.sort_values('emoji_count', ascending=False)
        
        # Create subplots
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=('Messages', 'Words', 'Media Shared', 'Emojis Used'),
            shared_xaxes=False
        )
        
        # Add bars for each metric
        fig.add_trace(go.Bar(x=user_messages['user'], y=user_messages['count'], name='Messages'), row=1, col=1)
        fig.add_trace(go.Bar(x=user_words['user'], y=user_words['word_count'], name='Words'), row=1, col=2)
        fig.add_trace(go.Bar(x=user_media['user'], y=user_media['has_media'], name='Media'), row=2, col=1)
        fig.add_trace(go.Bar(x=user_emojis['user'], y=user_emojis['emoji_count'], name='Emojis'), row=2, col=2)
        
        fig.update_layout(
            title_text='Top User Comparison',
            height=800,
            showlegend=False,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        return fig
        
    def get_conversation_by_date(self, date):
        """
        Get conversation for a specific date
        
        Args:
            date (datetime.date): Date to filter
            
        Returns:
            pd.DataFrame: Filtered DataFrame with messages from that date
        """
        if self.df.empty:
            return pd.DataFrame()
            
        # Convert date to datetime.date if it's not already
        if hasattr(date, 'date'):
            date = date.date()
            
        # Filter by date
        return self.df[self.df['date'].dt.date == date].copy()
        
    def cleanup_temp_data(self):
        """
        Remove temporary columns and clear memory
        Should be called periodically to maintain memory efficiency
        """
        # Remove temporary columns if they exist
        temp_columns = ['year_month_str']
        for col in temp_columns:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
        
        # Force garbage collection
        gc.collect()
        
    def __del__(self):
        """
        Destructor to ensure cleanup when instance is deleted
        """
        try:
            self.cleanup_temp_data()
        except:
            pass 