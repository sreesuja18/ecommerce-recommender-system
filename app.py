import streamlit as st

st.title("Product Recommendation System")

st.write("This is my ML recommendation app 🚀")

product_name = st.text_input("Enter Product Name")

if st.button("Recommend"):
    st.write("Recommendations will appear here")
