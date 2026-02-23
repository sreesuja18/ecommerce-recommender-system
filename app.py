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
    df = pd.read_csv("small_train.csv")  # <-- PUT YOUR EXACT CSV NAME
    df = df.fillna("")
    df["combined_features"] = df["TITLE"] + " " + df["BULLET_POINTS"] + " " + df["DESCRIPTION"]
    return df

df = load_data()

# -----------------------------
# TRENDING SECTION
# -----------------------------
st.subheader("🔥 Trending Products")
st.dataframe(df[['TITLE']].head(5))
st.markdown("---")

# -----------------------------
# VECTORIZE
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def get_recommendations(product_name, top_n=5):
    product_name = product_name.lower()

    matches = df[df["TITLE"].str.lower().str.contains(product_name)]

    if matches.empty:
        return None

    idx = matches.index[0]

    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    return similarity_scores[1:top_n+1]

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

                st.markdown("### " + df.loc[i, "TITLE"])
                st.write("Similarity Score:", round(score, 3))

                with st.expander("View Description"):
                    st.write(df.loc[i, "DESCRIPTION"])

                st.markdown("---")
