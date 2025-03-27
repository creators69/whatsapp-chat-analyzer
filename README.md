# WhatsApp Chat Analyzer

A powerful tool to analyze WhatsApp chat exports and generate insightful visualizations and statistics.

## Features

- Upload WhatsApp chat export (.txt file)
- View overall statistics (total messages, words, media, links)
- Analyze message activity by date, time, and user
- Generate word clouds of most common words
- Identify most active users in group chats
- View sentiment analysis of messages
- Examine emoji usage statistics
- Timeline analysis of chat activity

## Live Demo

Try the WhatsApp Chat Analyzer live at: https://whatsapp-chat-analyzer-yourusername.streamlit.app

## How to Use

1. **Export your WhatsApp chat**:
   - Open WhatsApp
   - Go to the chat you want to analyze
   - Tap the three dots (menu) > More > Export chat
   - Choose "Without Media" for faster processing
   - Save the .txt file

2. **Upload to the analyzer**:
   - Go to the [WhatsApp Chat Analyzer](https://whatsapp-chat-analyzer-yourusername.streamlit.app)
   - Upload your .txt chat file
   - View your personalized analytics

## Local Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/whatsapp-chat-analyzer.git
cd whatsapp-chat-analyzer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run the app
streamlit run app.py
```

## Deployment Options

### Streamlit Cloud (Easiest)

1. Fork this repository
2. Sign up at [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app pointing to your fork
4. Select app.py as the main file

### Other Deployment Options

See the deployment guides in the docs folder for more options:
- Docker deployment
- Heroku deployment
- AWS/GCP/Azure deployment
- Offline distribution

## Privacy

- All analysis is done in your browser
- No chat data is stored on any server
- Your data never leaves your computer when using the local version

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.