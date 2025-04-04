import streamlit as st
import pickle

with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

st.title("📩 Spam Email Detector")
st.write("Enter the email content below to check if it's spam or not.")

user_input = st.text_area("Email Content", height=200)

if st.button("Check Email"):
    if user_input.strip():
        input_features = vectorizer.transform([user_input])  # Convert text to feature vector
        prediction = model.predict(input_features)[0]

        if prediction == 1:
            st.success("✅ This is not a Spam email.")
        else:
            st.error("🚨 This is a Spam email!")

    else:
        st.warning("⚠️ Please enter some text.")

