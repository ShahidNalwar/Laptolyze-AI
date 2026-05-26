import streamlit as st
import re
from serpapi import GoogleSearch

SERPAPI_KEY = "6487a202693a533af5ff0382abdd78750c5b992fa3c8a2f18fd87aa27c02bcb9"  # ← paste your key here

LAPTOPS = {
    "Samsung Galaxy Book4 (Core i5, 16GB, 512GB)":          "NP750XGJ",
    "ASUS Zenbook 14 OLED (Core Ultra 5, 16GB, 512GB)":     "UX3405CA-PZ348WS",
    "Acer Predator Helios Neo 16 (i7, 16GB, 1TB)":          "PHN16-72-77GZ",
    "Lenovo LOQ 15 (i7 13th Gen, 16GB, 512GB)":             "15IRX9",
    "Acer Aspire 7 (Ryzen 5, 16GB, 512GB)":                 "A715-42G",
    "Asus TUF Gaming F15 (i5, 16GB, 512GB)":                "FX507ZC4-HN116W",
    "HP Omen 16 (Ryzen 9, 24GB, 1TB)":                      "ap0182AX",
    "Lenovo Legion 5 (Ryzen 7, 16GB, 512GB)":               "82JW00E2IN",
    "Acer Nitro V 16 (Ryzen 5, 16GB, 512GB)":               "ANV15-41",
    "MSI Katana A15 AI (Ryzen 7, 16GB, 512GB)":             "B8VE-481IN",
    "Asus Vivobook 16 (Core i5, 16GB, 512GB)":              "X1605VA-SH2124W",
    "HP Victus 15 (Ryzen 5, 16GB, 512GB)":                  "15-fb0106AX",
    "Dell G15 5530 (i5, 16GB, 1TB)":                        "OGN55301114P01RIN",
    "Lenovo IdeaPad Gaming 3 (Ryzen 5, 8GB, 512GB)":        "82K20277IN",
    "MacBook Air M3 (8GB, 256GB)":                           "MRYR3HN/A",
    "ASUS ROG Strix G16 (i9, 32GB, 1TB)":                   "G614JIR-N4055WS",
    "Samsung Galaxy Book5 Pro (Core Ultra 7, 16GB, 512GB)": "NP960XHA",
    "HP Pavilion Plus (Core Ultra 5, 16GB, 512GB)":         "14-ew1082TU",
    "Acer Swift Go 14 (Core Ultra 5, 16GB, 512GB)":         "SFG14-73",
    "Lenovo Yoga Slim 7 (Core Ultra 7, 16GB, 512GB)":       "14IMH9",
}

st.set_page_config(page_title="Live Laptop Price Hub", page_icon="💰", layout="centered")

st.html("""
<style>
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    .price-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px; border-radius: 16px; margin-bottom: 14px;
        display: flex; justify-content: space-between; align-items: center;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .price-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
    .seller-box  { display: flex; flex-direction: column; gap: 5px; }
    .seller-name { font-size: 1.1rem; font-weight: 600; color: #f1f5f9; }
    .seller-sub  {
        font-size: 0.76rem; color: #94a3b8; max-width: 320px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .best-badge {
        background: linear-gradient(90deg, #22c55e, #15803d);
        color: white; padding: 3px 10px; border-radius: 9999px;
        font-size: 0.68rem; font-weight: 700; width: fit-content;
    }
    .action-area { display: flex; align-items: center; gap: 18px; }
    .price-tag   { font-size: 1.45rem; font-weight: 800; color: #38bdf8; white-space: nowrap; }
    .buy-link-btn {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: #ffffff !important; padding: 9px 18px; border-radius: 10px;
        text-decoration: none !important; font-size: 0.88rem; font-weight: 700;
        box-shadow: 0 4px 12px rgba(37,99,235,0.2);
        display: inline-block; white-space: nowrap; transition: all 0.2s ease;
    }
    .buy-link-btn:hover { background: linear-gradient(135deg, #3b82f6, #2563eb); transform: scale(1.03); }
    .model-tag {
        font-size: 0.72rem; color: #38bdf8;
        background: rgba(56,189,248,0.08);
        border: 1px solid rgba(56,189,248,0.2);
        padding: 2px 8px; border-radius: 6px; font-family: monospace;
    }
</style>
""")

def extract_price(price_text: str) -> int | None:
    if not price_text:
        return None
    digits = re.sub(r"[^\d]", "", str(price_text))
    if digits and 4 <= len(digits) <= 7:
        return int(digits)
    return None

def _search_shopping(query: str) -> list[dict]:
    params = {
        "engine": "google_shopping", "q": query,
        "gl": "in", "hl": "en",
        "google_domain": "google.co.in",
        "num": "40", "api_key": SERPAPI_KEY,
    }
    try:
        api_results = GoogleSearch(params).get_dict()
    except Exception as e:
        st.error(f"SerpAPI Error: {e}")
        return []

    shopping = api_results.get("shopping_results", [])
    if not shopping:
        return []

    raw = []
    for item in shopping:
        seller = (item.get("source") or item.get("seller") or item.get("store") or "").strip()
        if not seller:
            continue
        price = extract_price(item.get("price", ""))
        if not price or price < 15000 or price > 800000:
            continue
        title = item.get("title", "")[:90]
        link  = (item.get("link") or item.get("product_link") or item.get("url") or "")
        if not link:
            link = f"https://www.google.com/search?q={query.replace(' ', '+')}+buy+india"
        raw.append({"seller": seller, "price": price, "title": title, "link": link})

    seen = {}
    for item in raw:
        key = item["seller"].lower().strip()
        if key not in seen or item["price"] < seen[key]["price"]:
            seen[key] = item

    return sorted(seen.values(), key=lambda x: x["price"])

def fetch_live_prices(model_number: str, display_name: str) -> list[dict]:
    results = _search_shopping(model_number)
    if not results:
        st.caption("Model number returned 0 results — trying name-based fallback...")
        results = _search_shopping(display_name)
    return results

# ── UI ──
st.title("💰 Live Laptop Price Tracker")
st.write("Real-time prices fetched using exact model numbers from Google Shopping India.")

with st.container():
    laptop_choice = st.selectbox("Select a Laptop Model", list(LAPTOPS.keys()))
    model_number  = LAPTOPS[laptop_choice]
    st.markdown(f"Model number: <span class='model-tag'>{model_number}</span>", unsafe_allow_html=True)
    fetch_trigger = st.button("🔍 Fetch Live Prices", use_container_width=True, type="primary")

st.markdown("---")

if fetch_trigger:
    with st.spinner(f"Searching for **{model_number}** across Indian retailers..."):
        offers = fetch_live_prices(model_number, laptop_choice)

    if not offers:
        st.warning(
            f"No results found for model `{model_number}`.\n\n"
            "Possible reasons:\n"
            "- Model number not indexed on Google Shopping India\n"
            "- Product out of stock across all platforms\n"
            "- SerpAPI free quota (100/month) exhausted\n\n"
            f"[Search manually]"
            f"(https://www.google.com/search?q={model_number}+laptop+buy+india&tbm=shop)"
        )
    else:
        lowest  = offers[0]["price"]
        highest = offers[-1]["price"]
        savings = highest - lowest

        c1, c2, c3 = st.columns(3)
        c1.metric("Lowest Price",  f"₹{lowest:,}")
        c2.metric("Highest Price", f"₹{highest:,}")
        c3.metric("Max Savings",   f"₹{savings:,}")

        st.markdown(f"### {len(offers)} seller(s) found for `{model_number}`")
        st.write("")

        for offer in offers:
            is_best    = offer["price"] == lowest
            badge_html = '<div class="best-badge">✓ BEST DEAL</div>' if is_best else ""
            st.html(f"""
            <div class="price-card">
                <div class="seller-box">
                    <div class="seller-name">{offer['seller']}</div>
                    <div class="seller-sub">{offer['title']}</div>
                    {badge_html}
                </div>
                <div class="action-area">
                    <div class="price-tag">₹{offer['price']:,}</div>
                    <a href="{offer['link']}" target="_blank" class="buy-link-btn">Buy ↗</a>
                </div>
            </div>
            """)

        if savings > 0:
            st.success(
                f"💡 Best deal on **{offers[0]['seller']}** — "
                f"saves ₹{savings:,} vs {offers[-1]['seller']} (₹{highest:,})"
            )
else:
    st.info("👆 Select a laptop and click **Fetch Live Prices** to compare exact-model deals.")