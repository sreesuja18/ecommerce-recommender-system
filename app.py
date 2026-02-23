import streamlit as st
import pandas as pd
st.set_page_config(page_title="Ecommerce Recommender", layout="wide")

st.title("🛍️ Ecommerce Product Recommendation System")
st.markdown("---")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🛍️ Product Recommendation System")

st.markdown("""
This app recommends similar products using **TF-IDF** and **Cosine Similarity**.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("small_train.csv")
    df['text'] = df['TITLE'].fillna('') + " " + \
                 df['BULLET_POINTS'].fillna('') + " " + \
                 df['DESCRIPTION'].fillna('')
    return df

df = load_data()

# Compute similarity
@st.cache_resource
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data['text'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_similarity(df)

# Recommendation function
def recommend(product_title, num_recommendations=5):
    matches = df[df['TITLE'].str.contains(product_title, case=False, na=False)]

    if matches.empty:
        return None

    idx = matches.index[0]

    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similarity_scores = similarity_scores[1:num_recommendations+1]

    product_indices = [i[0] for i in similarity_scores]

    return df[['PRODUCT_ID', 'TITLE']].iloc[product_indices]

# UI
st.subheader("🔥 Trending Products")

st.dataframe(df[['TITLE','PRICE']].head(5))
user_input = st.text_input("Enter Product Name")

if st.button("Recommend"):
    for i, score in recommendations:
    col1, col2 = st.columns([1,3])

    with col1:
        st.image(df.loc[i, 'IMAGE_URL'], width=120)

    with col2:
        st.subheader(df.loc[i, 'TITLE'])
        st.write("💰 Price: ₹", df.loc[i, 'PRICE'])
        st.write("Similarity Score:", round(score, 3))

        if st.button(f"Add to Cart {i}"):
            st.success("Added to cart!")
            

