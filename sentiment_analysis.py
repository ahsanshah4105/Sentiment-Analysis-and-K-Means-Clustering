import os
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Download necessary NLTK resources
nltk.download('vader_lexicon', quiet=True)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def perform_sentiment_analysis(text):
    sentiment = sia.polarity_scores(text)
    return sentiment

def get_sentiment_name(compound_score):
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'reviewText' not in df.columns:
            raise ValueError("The file does not contain a 'reviewText' column.")
        return df['reviewText'].astype(str)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud for Product Reviews")
    plt.show()

def calculate_sentiment_summary(sentiment_results):
    positive = sentiment_results.count("Positive")
    negative = sentiment_results.count("Negative")
    neutral = sentiment_results.count("Neutral")

    total = len(sentiment_results)
    avg_sentiment = (positive - negative) / total

    rating = (
        "★★★★★" if avg_sentiment >= 0.5 else
        "★★★★" if avg_sentiment >= 0.2 else
        "★★★" if avg_sentiment >= 0 else
        "★★" if avg_sentiment >= -0.2 else
        "★"
    )

    return {
        "Star Rating": rating,
        "Positive": positive,
        "Negative": negative,
        "Neutral": neutral,
        "Positive Ratio": positive / total,
        "Negative Ratio": negative / total,
        "Neutral Ratio": neutral / total,
        "Average Sentiment": avg_sentiment
    }

def perform_kmeans_clustering(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    return kmeans.labels_, X

if __name__ == "__main__":
    file_path = r"E:\reviews\reviews_500.csv"  # Ensure the CSV file is in the same directory or provide full path

    reviews = read_csv_file(file_path)

    if reviews is None:
        print("Failed to read the file or 'reviewText' column is missing. Exiting program.")
        exit()

    print(f"Loaded {len(reviews)} reviews.")

    # Generate Word Cloud
    combined_text = " ".join(reviews)
    generate_word_cloud(combined_text)

    # Sentiment Analysis
    sentiment_results = []
    sentiment_data = []

    print("Starting sentiment analysis...")

    for i, text in enumerate(reviews, start=1):
        sentiment = perform_sentiment_analysis(text)
        sentiment_label = get_sentiment_name(sentiment['compound'])

        sentiment_results.append(sentiment_label)
        sentiment_data.append({
            'Feedback': text,
            'Sentiment': sentiment_label
        })


    # K-Means Clustering
    cluster_labels, tfidf_matrix = perform_kmeans_clustering(reviews)

    # PCA Visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(tfidf_matrix.toarray())

    plt.figure(figsize=(10, 5))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis')
    plt.title("K-Means Clustering of Reviews (PCA reduced to 2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Cluster')
    plt.show()

    # Print Sentiment Summary
    summary = calculate_sentiment_summary(sentiment_results)

    print("\n" + "="*40)
    print("          SENTIMENT SUMMARY")
    print("="*40)
    print(f"Star Rating         : {summary['Star Rating']}")
    print(f"Positive Reviews    : {summary['Positive']} ({summary['Positive Ratio']*100:.2f}%)")
    print(f"Negative Reviews    : {summary['Negative']} ({summary['Negative Ratio']*100:.2f}%)")
    print(f"Neutral Reviews     : {summary['Neutral']} ({summary['Neutral Ratio']*100:.2f}%)")
    print(f"Average Sentiment   : {summary['Average Sentiment']:.2f}")
    print("="*40)

    # Save Sentiment Results
    pd.DataFrame(sentiment_data).to_csv('sentiment_analysis_results.csv', index=False)
    print("Sentiment analysis results saved to 'sentiment_analysis_results.csv'")
