#!/bin/bash

# Make directory for NLTK data
mkdir -p ~/.nltk_data

# Download NLTK data
python -m nltk.downloader punkt
python -m nltk.downloader stopwords

echo "NLTK data downloaded successfully" 