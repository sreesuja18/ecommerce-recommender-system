import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Ecommerce Recommender", layout="wide")

st.title("🛍️ Ecommerce Product Recommendation System")
st.markdown("---")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")   # <-- Make sure your CSV name is correct
    df = df.dropna(subset=['TITLE'])
    return df

df = load_data()

# -----------------------------
# TRENDING SECTION
# -----------------------------
st.subheader("🔥 Trending Products")

if "PRICE" in df.columns:
    st.dataframe(df[['TITLE', 'PRICE']].head(5))
else:
    st.dataframe(df[['TITLE']].head(5))

st.markdown("---")

# -----------------------------
# RECOMMENDATION LOGIC
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['TITLE'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(product_name, top_n=5):
    product_name = product_name.lower()

    matches = df[df['TITLE'].str.lower().str.contains(product_name)]

    if matches.empty:
        return None

    idx = matches.index[0]

    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_products = similarity_scores[1:top_n+1]

    return top_products

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("🔍 Find Similar Products")

user_input = st.text_input("Enter Product Name")

if st.button("Recommend"):

    if user_input.strip() == "":
        st.warning("Please enter a product name.")
    else:
        recommendations = get_recommendations(user_input)

        if recommendations is None:
            st.warning("Product not found. Please check spelling.")
        else:
            st.success("Top Recommendations")

            for i, score in recommendations:

                col1, col2 = st.columns([1, 3])

                with col1:
                    if "IMAGE_URL" in df.columns:
                        st.image(df.loc[i, 'IMAGE_URL'], width=120)
                    else:
                        st.write("🖼️ No Image")

                with col2:
                    st.subheader(df.loc[i, 'TITLE'])

                    if "PRICE" in df.columns:
                        st.write("💰 Price: ₹", df.loc[i, 'PRICE'])

                    st.write("Similarity Score:", round(score, 3))

                    if st.button(f"Add to Cart {i}"):
                        st.success("Added to cart!")
