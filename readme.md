# Laptolyze AI

An all-in-one laptop research platform that combines BERT-based
sentiment analysis on product reviews with live price tracking
across multiple e-commerce platforms using SerpAPI.

## Problem

Buying a laptop is overwhelming — reviews are scattered across
platforms, prices change daily, and sentiment is hard to gauge
from thousands of mixed reviews.

## Solution

Laptolyze aggregates laptop reviews, runs BERT sentiment classification
to give a clear positive/negative/neutral breakdown, and tracks live
prices across Amazon, Flipkart, and other platforms in real time.

## Tech Stack

| Layer | Technology |
|---|---|
| NLP Model | HuggingFace BERT (fine-tuned) |
| Price Tracking | SerpAPI |
| Backend | Python, FastAPI |
| Frontend | Streamlit (pages/) |
| ML Tracking | MLflow |
| Data | Pandas, Scikit-learn |
| Serialization | Pickle (label_encoder.pkl) |

## Architecture

## Architecture

```
                    ┌─────────────────────┐
                    │   User Interface     │
                    │  (Streamlit app.py) │
                    └──────────┬──────────┘
                               │
                    User enters laptop name
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│      Price Tracker      │       │     Review Fetcher      │
│                         │       │                         │
│  SerpAPI fetches live   │       │  fetch_reviews.py       │
│  prices from:           │       │  scrapes reviews from   │
│  • Amazon               │       │  multiple platforms     │
│  • Flipkart             │       │                         │
│  • Other platforms      │       └────────────┬────────────┘
└────────────┬────────────┘                    │
             │                                 ▼
             │                  ┌─────────────────────────┐
             │                  │    BERT Classifier      │
             │                  │                         │
             │                  │  train_model.py         │
             │                  │  Fine-tuned on laptop   │
             │                  │  reviews corpus         │
             │                  │                         │
             │                  │  Output:                │
             │                  │  ✅ Positive            │
             │                  │  ❌ Negative            │
             │                  │  ➖ Neutral             │
             └──────────────────┴─────────────┐
                                              │
                                              ▼
                               ┌─────────────────────────┐
                               │    Streamlit Dashboard  │
                               │                         │
                               │  • Live price comparison│
                               │  • Sentiment breakdown  │
                               │  • Confidence scores    │
                               │  • Buy recommendation   │
                               └─────────────────────────┘
```

## Key Features

- BERT-based sentiment classifier fine-tuned on laptop reviews
- Live price tracking via SerpAPI (real-time, not scraped static data)
- MLflow experiment tracking for model versioning
- Multi-page Streamlit interface
- Label encoding for multi-class sentiment output

## Design Tradeoffs

- SerpAPI has rate limits on free tier — implemented response caching
  to avoid redundant calls for the same laptop model
- BERT inference is slow — used smaller distilBERT variant for
  production speed vs accuracy tradeoff
- Chose Streamlit over React for faster development iteration

## What I Would Do Differently

- Add a spec comparison engine (RAM, GPU, battery side by side)
- Build a personalized recommendation engine based on use case
  (gaming, coding, design)
- Replace SerpAPI with direct retailer APIs for better reliability
- Add price history charts with trend prediction

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your SerpAPI key to environment variables before running.

## File Structure

## File Structure

```
Laptolyze-AI/
│
├── app.py                  # Main Streamlit app entry point
├── train_model.py          # BERT fine-tuning and model training
├── fetch_reviews.py        # Review scraping from e-commerce platforms
├── label_encoder.pkl       # Saved label encoder (pos/neg/neutral)
├── mlflow.db               # MLflow experiment tracking database
├── requirements.txt        # Python dependencies
│
├── data/                   # Training data directory
│   └── reviews.csv         # Labelled laptop reviews dataset
│
└── pages/                  # Multi-page Streamlit pages
    ├── price_tracker.py    # Live price comparison page
    ├── sentiment.py        # Sentiment analysis results page
    └── compare.py          # Laptop comparison page
```
