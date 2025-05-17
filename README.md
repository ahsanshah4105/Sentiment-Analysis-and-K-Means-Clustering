# Sentiment Analysis of Product Reviews

This project analyzes product reviews from a CSV file to determine whether each review is positive, negative, or neutral. It also generates a word cloud of common words in the reviews and groups similar reviews using machine learning clustering.

## Features

- Reads product reviews from a CSV file
- Performs sentiment analysis on each review using NLTK’s VADER
- Generates a word cloud to visualize frequently used words
- Clusters similar reviews with K-Means clustering and visualizes results in 2D
- Summarizes sentiment results with star ratings and percentages
- Saves sentiment analysis results to a CSV file

## How to use

1. Place your CSV file containing product reviews in the `reviewText` column.
2. Update the file path in the script (`file_path` variable) to point to your CSV file.
3. Run the Python script:
   ```bash
   python your_script_name.py


