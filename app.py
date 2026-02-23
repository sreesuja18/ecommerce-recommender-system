import streamlit as st
import pandas as pd
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
user_input = st.text_input("Enter Product Name")

if st.button("Recommend"):
    if user_input.strip() == "":
        st.warning("Please enter a product name.")
    else:
        recommendations = recommend(user_input)

        if recommendations is None:
            st.error("❌ No matching product found.")
        else:
            st.success("Top Recommendations:")
            st.dataframe(recommendations)
