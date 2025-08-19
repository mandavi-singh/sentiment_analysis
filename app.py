import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model and vectorizer
model = pickle.load(open("sentiment_model_xgb_augmented.pkl", "rb"))
vec = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Load dataset
data = pd.read_csv("3) Sentiment dataset.csv").dropna(subset=["Text"])

# Dropdown for dataset text
texts = data["Text"].unique().tolist()

st.title("📊 Sentiment Analysis App")
st.write("Select a text from the dataset to analyze its sentiment.")

user_input = st.selectbox("Choose a text:", texts)

if st.button("Predict"):
    # Fetch row for selected text
    row = data[data["Text"] == user_input].iloc[0]

    # Show dataset info
    st.subheader("📌 Original Information from Dataset")
    st.write(f"**User:** {row.get('User', 'N/A')}")
    st.write(f"**Platform:** {row.get('Platform', 'N/A')}")
    st.write(f"**Hashtags:** {row.get('Hashtags', 'N/A')}")
    st.write(f"**Country:** {row.get('Country', 'N/A')}")
    st.write(f"**Likes:** {row.get('Likes', 'N/A')}")
    st.write(f"**Year:** {row.get('Year', 'N/A')}")
    st.write(f"**True Sentiment (from dataset):** {row.get('Sentiment', 'N/A')}")

    # Model prediction
    vec_input = vec.transform([user_input])
    prediction = model.predict(vec_input)[0]
    prob = model.predict_proba(vec_input)[0]

    st.subheader("🤖 Model Prediction")
    if prediction == 1:
        st.success(f"Positive Sentiment ✅ (Confidence: {prob[1]*100:.2f}%)")
    elif prediction == 0:
        st.error(f"Negative Sentiment ❌ (Confidence: {prob[0]*100:.2f}%)")
    else:
        st.info(f"Neutral Sentiment ⚪ (Confidence: {prob[2]*100:.2f}%)")

    # Show probabilities
    st.subheader("📊 Prediction Probabilities")
    st.write(f"🔴 Negative: {prob[0]*100:.2f}%")
    st.write(f"🟢 Positive: {prob[1]*100:.2f}%")
    st.write(f"⚪ Neutral: {prob[2]*100:.2f}%")

# =====================
# 📈 Visualization Section
# =====================
st.subheader("📊 Dataset Insights")

# Filter controls
year_filter = st.selectbox("📅 Select Year (optional)", ["All"] + sorted(data["Year"].dropna().unique().tolist()))
country_filter = st.selectbox("🌍 Select Country (optional)", ["All"] + sorted(data["Country"].dropna().unique().tolist()))

filtered_data = data.copy()
if year_filter != "All":
    filtered_data = filtered_data[filtered_data["Year"] == year_filter]
if country_filter != "All":
    filtered_data = filtered_data[filtered_data["Country"] == country_filter]

# Sentiment distribution (filtered)
fig, ax = plt.subplots()
filtered_data['Sentiment'].value_counts().plot(kind="bar", ax=ax, color=["red","green","gray"])
ax.set_title("Sentiment Distribution")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
st.pyplot(fig)

# Top 5 Users by Likes
if "Likes" in filtered_data.columns:
    fig2, ax2 = plt.subplots()
    filtered_data.groupby("User")["Likes"].sum().nlargest(5).plot(kind="bar", ax=ax2, color="blue")
    ax2.set_title("Top 5 Users by Likes")
    ax2.set_ylabel("Total Likes")
    st.pyplot(fig2)