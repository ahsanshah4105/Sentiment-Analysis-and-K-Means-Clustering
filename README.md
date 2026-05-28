# 🧠 Sentiment Analysis & K-Means Clustering of Product Reviews

> A Python NLP project that analyzes product reviews using VADER sentiment analysis, visualizes word frequency via word clouds, and groups similar reviews using K-Means clustering with PCA dimensionality reduction.

![Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python)
![NLTK](https://img.shields.io/badge/NLP-NLTK%20VADER-blue)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)
![Purpose](https://img.shields.io/badge/Purpose-Learning%20%2F%20Demo-yellow)

---

## 📊 Output Screenshots

<p align="center">
  <img src="https://github.com/user-attachments/assets/f22faf99-e0a2-449b-a952-2ccb3796fc04" width="700" alt="Word Cloud"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/752b51e3-5dff-4c40-a560-70962659ea1c" width="700" alt="K-Means Cluster Plot"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/178b52b3-9c97-467b-8253-c3a1444b30c8" width="700" alt="Sentiment Summary Output"/>
</p>

---

## ✨ Features

- 📥 **CSV ingestion** — reads product reviews from a `reviewText` column
- 💬 **Sentiment classification** — labels each review as Positive, Negative, or Neutral using NLTK VADER
- ☁️ **Word Cloud** — visualizes the most frequently used words across all reviews
- 🤖 **K-Means Clustering** — groups similar reviews into 3 clusters using TF-IDF vectorization
- 📉 **PCA Visualization** — reduces high-dimensional clusters to 2D scatter plot
- ⭐ **Star Rating Summary** — calculates overall rating (1–5 stars) from sentiment ratios
- 💾 **CSV export** — saves per-review sentiment labels to `sentiment_analysis_results.csv`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| NLTK (VADER) | Rule-based sentiment scoring |
| scikit-learn | TF-IDF vectorization + K-Means clustering |
| PCA | Dimensionality reduction for 2D visualization |
| WordCloud | Visual frequency map of review text |
| Matplotlib | Plotting cluster scatter + word cloud |
| Pandas | CSV reading and data manipulation |

---

## 🔍 How It Works

```
CSV File (reviewText column)
        │
        ▼
 Word Cloud Generation
        │
        ▼
 VADER Sentiment Scoring
  → Positive (compound ≥ 0.05)
  → Negative (compound ≤ -0.05)
  → Neutral  (between)
        │
        ▼
 TF-IDF Vectorization
        │
        ▼
 K-Means Clustering (k=3)
        │
        ▼
 PCA → 2D Scatter Plot
        │
        ▼
 Star Rating Summary + CSV Export
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ahsanshah4105/Sentiment-Analysis-and-K-Means-Clustering.git
cd Sentiment-Analysis-and-K-Means-Clustering
```

### 2. Install dependencies
```bash
pip install nltk pandas scikit-learn wordcloud matplotlib
```

### 3. Prepare your dataset
Your CSV file must have a column named `reviewText`. Example:

| reviewText |
|---|
| "This product is amazing, highly recommend!" |
| "Terrible quality, broke after one day." |
| "It's okay, nothing special." |

### 4. Update the file path
In the script, update this line to point to your CSV:
```python
file_path = r"path/to/your/reviews.csv"
```

### 5. Run the script
```bash
python sentiment_analysis.py
```

---

## 📤 Sample Output

```
========================================
          SENTIMENT SUMMARY
========================================
Star Rating         : ★★★★
Positive Reviews    : 312 (62.40%)
Negative Reviews    : 88  (17.60%)
Neutral Reviews     : 100 (20.00%)
Average Sentiment   : 0.25
========================================
Sentiment analysis results saved to 'sentiment_analysis_results.csv'
```

---

## 📞 Contact

**Ahsan Ali Shah** — Mobile & Python Developer  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/ahsanalishah4105)[![Email](https://img.shields.io/badge/Email-ahsanalishah4105@gmail.com-red?logo=gmail)](mailto:ahsanalishah4105@gmail.com)
