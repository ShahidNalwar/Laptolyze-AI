"""
fetch_reviews.py
────────────────
Fetches real user reviews for laptops using SerpAPI (Google Search)
and stores them in a CSV file ready for sentiment analysis / ML pipelines.

Usage:
    python fetch_reviews.py

Requirements:
    pip install google-search-results pandas
"""

import time
import re
import os
import pandas as pd
from serpapi import GoogleSearch


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SERPAPI_KEY = "6487a202693a533af5ff0382abdd78750c5b992fa3c8a2f18fd87aa27c02bcb9"

OUTPUT_FILE = "data/laptop_reviews.csv"

REVIEWS_PER_LAPTOP = 50

DELAY_BETWEEN_LAPTOPS = 2


# ─────────────────────────────────────────────────────────────────────────────
# LAPTOP LIST
# ─────────────────────────────────────────────────────────────────────────────

LAPTOPS = [
    "ASUS Zenbook 14 OLED",
    "Samsung Galaxy Book4",
    "Acer Predator Helios Neo 16",
    "Lenovo LOQ 13th Gen",
    "Acer Aspire 7",
    "Asus TUF Gaming F15",
    "HP Omen 16",
    "Lenovo Legion 5",
    "Acer Nitro V 16",
    "MSI Katana A15",
    "Asus Vivobook 16",
    "HP Victus",
    "Dell G15",
    "Lenovo IdeaPad Gaming 3",
    "MacBook Air M3",
    "ASUS ROG Strix G16",
    "Samsung Galaxy Book5 Pro",
    "HP Pavilion Plus",
    "Acer Swift Go 14",
    "Lenovo Yoga Slim 7"
]


# ─────────────────────────────────────────────────────────────────────────────
# LAPTOP SPECS
# ─────────────────────────────────────────────────────────────────────────────

LAPTOP_SPECS = {

    "ASUS Zenbook 14 OLED": {
        "processor": "Intel Core Ultra 7",
        "ram": "16GB",
        "storage": "1TB SSD",
        "display": 14.0,
        "os": "Windows 11",
        "price": 99990,
        "spec_rating": 5
    },

    "Samsung Galaxy Book4": {
        "processor": "Intel Core 7",
        "ram": "16GB",
        "storage": "512GB SSD",
        "display": 15.6,
        "os": "Windows 11",
        "price": 74990,
        "spec_rating": 4
    },

    "Acer Predator Helios Neo 16": {
        "processor": "Intel Core i7 14th Gen",
        "ram": "16GB",
        "storage": "1TB SSD",
        "display": 16.0,
        "os": "Windows 11",
        "price": 134990,
        "spec_rating": 5
    },

    "Lenovo LOQ 13th Gen": {
        "processor": "Intel Core i5 13th Gen",
        "ram": "16GB",
        "storage": "512GB SSD",
        "display": 15.6,
        "os": "Windows 11",
        "price": 72990,
        "spec_rating": 4
    },

    "Acer Aspire 7": {
        "processor": "AMD Ryzen 5",
        "ram": "16GB",
        "storage": "512GB SSD",
        "display": 15.6,
        "os": "Windows 11",
        "price": 55990,
        "spec_rating": 3
    },

    "Asus TUF Gaming F15": {
        "processor": "Intel Core i7 13th Gen",
        "ram": "16GB",
        "storage": "1TB SSD",
        "display": 15.6,
        "os": "Windows 11",
        "price": 89990,
        "spec_rating": 4
    },

    "HP Omen 16": {
        "processor": "Intel Core i7 14th Gen",
        "ram": "16GB",
        "storage": "1TB SSD",
        "display": 16.1,
        "os": "Windows 11",
        "price": 139990,
        "spec_rating": 5
    },

    "Lenovo Legion 5": {
        "processor": "AMD Ryzen 7",
        "ram": "16GB",
        "storage": "1TB SSD",
        "display": 16.0,
        "os": "Windows 11",
        "price": 129990,
        "spec_rating": 5
    },

    "Acer Nitro V 16": {
        "processor": "AMD Ryzen 7",
        "ram": "16GB",
        "storage": "512GB SSD",
        "display": 16.0,
        "os": "Windows 11",
        "price": 79990,
        "spec_rating": 4
    },

    "MSI Katana A15": {
        "processor": "AMD Ryzen 7",
        "ram": "16GB",
        "storage": "1TB SSD",
        "display": 15.6,
        "os": "Windows 11",
        "price": 94990,
        "spec_rating": 4
    },

    "Asus Vivobook 16": {
        "processor": "Intel Core i5 13th Gen",
        "ram": "16GB",
        "storage": "512GB SSD",
        "display": 16.0,
        "os": "Windows 11",
        "price": 64990,
        "spec_rating": 3
    },

    "HP Victus": {
        "processor": "AMD Ryzen 5",
        "ram": "16GB",
        "storage": "512GB SSD",
        "display": 15.6,
        "os": "Windows 11",
        "price": 68990,
        "spec_rating": 4
    },

    "Dell G15": {
        "processor": "Intel Core i5 13th Gen",
        "ram": "16GB",
        "storage": "512GB SSD",
        "display": 15.6,
        "os": "Windows 11",
        "price": 79990,
        "spec_rating": 4
    },

    "Lenovo IdeaPad Gaming 3": {
        "processor": "AMD Ryzen 5",
        "ram": "16GB",
        "storage": "512GB SSD",
        "display": 15.6,
        "os": "Windows 11",
        "price": 65990,
        "spec_rating": 4
    },

    "MacBook Air M3": {
        "processor": "Apple M3",
        "ram": "16GB",
        "storage": "512GB SSD",
        "display": 13.6,
        "os": "macOS",
        "price": 134900,
        "spec_rating": 5
    },

    "ASUS ROG Strix G16": {
        "processor": "Intel Core i9 14th Gen",
        "ram": "32GB",
        "storage": "1TB SSD",
        "display": 16.0,
        "os": "Windows 11",
        "price": 189990,
        "spec_rating": 5
    },

    "Samsung Galaxy Book5 Pro": {
        "processor": "Intel Core Ultra 7",
        "ram": "16GB",
        "storage": "1TB SSD",
        "display": 16.0,
        "os": "Windows 11",
        "price": 129990,
        "spec_rating": 5
    },

    "HP Pavilion Plus": {
        "processor": "Intel Core Ultra 5",
        "ram": "16GB",
        "storage": "512GB SSD",
        "display": 14.0,
        "os": "Windows 11",
        "price": 89990,
        "spec_rating": 4
    },

    "Acer Swift Go 14": {
        "processor": "Intel Core Ultra 7",
        "ram": "16GB",
        "storage": "512GB SSD",
        "display": 14.0,
        "os": "Windows 11",
        "price": 79990,
        "spec_rating": 4
    },

    "Lenovo Yoga Slim 7": {
        "processor": "Intel Core Ultra 7",
        "ram": "16GB",
        "storage": "1TB SSD",
        "display": 14.0,
        "os": "Windows 11",
        "price": 104990,
        "spec_rating": 5
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean review text."""

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    text = text.encode(
        "ascii",
        errors="ignore"
    ).decode()

    return text.strip()


def estimate_rating(snippet: str) -> int:
    """
    Estimate rating from review text.
    """

    text = snippet.lower()

    positives = [
        "excellent",
        "great",
        "amazing",
        "love",
        "perfect",
        "best",
        "fantastic",
        "outstanding",
        "superb",
        "highly recommend",
        "impressive",
        "smooth",
        "fast",
        "powerful",
        "lightweight",
        "beautiful",
        "stunning",
        "worth",
        "good battery",
        "long battery"
    ]

    negatives = [
        "bad",
        "terrible",
        "horrible",
        "awful",
        "worst",
        "poor",
        "disappointed",
        "slow",
        "heating",
        "overheat",
        "issue",
        "problem",
        "defect",
        "return",
        "waste",
        "expensive",
        "not worth",
        "avoid",
        "broken",
        "crash",
        "lag"
    ]

    pos_count = sum(1 for word in positives if word in text)
    neg_count = sum(1 for word in negatives if word in text)

    if neg_count >= 2:
        return 1

    elif neg_count == 1 and pos_count == 0:
        return 2

    elif pos_count == 0 and neg_count == 0:
        return 3

    elif pos_count == 1:
        return 4

    else:
        return 5


# ─────────────────────────────────────────────────────────────────────────────
# FETCH REVIEWS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_reviews_for_laptop(
    laptop_name: str,
    api_key: str,
    count: int = 50
):

    all_snippets = []

    queries = [

        f"{laptop_name} Amazon reviews",

        f"{laptop_name} Flipkart user reviews",

        f"{laptop_name} Reddit user experience",

        f"{laptop_name} laptop review 2025",

        f"{laptop_name} pros cons heating battery performance",
    ]

    for query in queries:

        if len(all_snippets) >= count:
            break

        params = {
            "engine": "google",
            "q": query,
            "gl": "in",
            "hl": "en",
            "num": "10",
            "api_key": api_key,
        }

        try:

            search = GoogleSearch(params)

            results = search.get_dict()

            organic_results = results.get(
                "organic_results",
                []
            )

            for result in organic_results:

                snippet = clean_text(
                    result.get("snippet", "")
                )

                title = clean_text(
                    result.get("title", "")
                )

                if len(snippet) > 30:

                    all_snippets.append({

                        "review_title": title[:120],

                        "review_text": snippet,
                    })

            related_questions = results.get(
                "related_questions",
                []
            )

            for question in related_questions:

                answer = clean_text(

                    question.get("snippet", "") or
                    question.get("answer", "")
                )

                if len(answer) > 30:

                    all_snippets.append({

                        "review_title": clean_text(
                            question.get(
                                "question",
                                "User Review"
                            )
                        )[:120],

                        "review_text": answer,
                    })

            time.sleep(0.5)

        except Exception as e:

            print(f"\n⚠️ Error for query '{query}'")
            print(e)

            continue

    return all_snippets[:count]


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATASET ROW
# ─────────────────────────────────────────────────────────────────────────────

def build_row(
    laptop_name: str,
    review: dict
):

    specs = LAPTOP_SPECS.get(
        laptop_name,
        {}
    )

    rating = estimate_rating(
        review["review_text"]
    )

    return {

        "product_name": laptop_name,

        "full_name_in_reviews": laptop_name,

        "price(in Rs.)": specs.get(
            "price",
            69990
        ),

        "processor": specs.get(
            "processor",
            "Intel Core i5"
        ),

        "ram": specs.get(
            "ram",
            "16GB"
        ),

        "os": specs.get(
            "os",
            "Windows 11"
        ),

        "storage": specs.get(
            "storage",
            "512GB SSD"
        ),

        "display(in inch)": specs.get(
            "display",
            15.6
        ),

        "spec_rating": specs.get(
            "spec_rating",
            3
        ),

        "review_title": review["review_title"],

        "review_text": review["review_text"],

        "review_rating": rating,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    print("=" * 70)
    print("LAPTOP REVIEW FETCHER")
    print("Using SerpAPI + Google Search")
    print("=" * 70)

    if SERPAPI_KEY == "YOUR_SERPAPI_KEY_HERE":

        print("\n❌ ERROR:")
        print("Please add your SerpAPI key first.\n")

        return

    all_rows = []

    for index, laptop in enumerate(LAPTOPS, start=1):

        print(f"\n[{index}/{len(LAPTOPS)}] Fetching reviews for:")
        print(f"→ {laptop}")

        reviews = fetch_reviews_for_laptop(
            laptop,
            SERPAPI_KEY,
            REVIEWS_PER_LAPTOP
        )

        print(f"✅ Collected {len(reviews)} reviews")

        for review in reviews:

            row = build_row(
                laptop,
                review
            )

            all_rows.append(row)

        if index < len(LAPTOPS):

            time.sleep(
                DELAY_BETWEEN_LAPTOPS
            )

    # SAVE CSV

    if not all_rows:

        print("\n❌ No reviews collected.")
        print("Check your SerpAPI key.")

        return

    df = pd.DataFrame(all_rows)

    columns = [

        "product_name",

        "full_name_in_reviews",

        "price(in Rs.)",

        "processor",

        "ram",

        "os",

        "storage",

        "display(in inch)",

        "spec_rating",

        "review_title",

        "review_text",

        "review_rating"
    ]

    df = df[columns]

    os.makedirs(
        "data",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)

    print(f"✅ DATASET SAVED")

    print(f"\n📁 File:")
    print(OUTPUT_FILE)

    print(f"\n💻 Laptops covered:")
    print(df['product_name'].nunique())

    print(f"\n📝 Total reviews:")
    print(len(df))

    print(f"\n⭐ Average reviews per laptop:")
    print(round(
        len(df) / df['product_name'].nunique(),
        1
    ))

    print("\n⭐ Rating distribution:")

    print(
        df["review_rating"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("=" * 70)


if __name__ == "__main__":
    main()