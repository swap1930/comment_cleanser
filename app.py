import streamlit as st
import pandas as pd
import os
import csv
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import random

# Label mapping
label_map = {0: "Hate Speech 😠", 1: "Offensive 😡", 2: "Clean 😊"}

# Save uncertain inputs
def save_unknown_comment(text, probs):
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/new_data.csv", mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([text, probs[0], probs[1], probs[2]])

# Load dataset from feedback only
@st.cache_data(show_spinner="📥 Loading feedback dataset...")
def load_feedback_data():
    if not os.path.exists("feedback/new_data.csv"):
        return pd.DataFrame(columns=["tweet", "class"])

    df = pd.read_csv("feedback/new_data.csv", header=None,
                     names=["tweet", "hate", "offensive", "clean"])
    df["class"] = df[["hate", "offensive", "clean"]].astype(float).idxmax(axis=1).map({
        "hate": 0, "offensive": 1, "clean": 2
    })
    return df[["tweet", "class"]].dropna()

# Load clean suggestions
@st.cache_data
def load_positive_data():
    pos_df = pd.read_csv("data/positive_data.csv").dropna()
    pos_df["text"] = pos_df["text"].apply(lambda x: re.sub(r"\s*\[\d+\]$", "", str(x)).strip())
    pos_df = pos_df[pos_df["text"].apply(lambda x: isinstance(x, str) and len(x.split()) > 4)]
    pos_df = pos_df.drop_duplicates(subset="text").reset_index(drop=True)
    return pos_df["text"].tolist()

# Suggest clean comments based on similarity
def get_clean_suggestions(user_input, model, n=3):
    clean_texts = load_positive_data()
    if not clean_texts:
        return ["No suggestions available."]
    vectorizer = model.named_steps["tfidf"]
    input_vec = vectorizer.transform([user_input])
    clean_vecs = vectorizer.transform(clean_texts)
    sims = cosine_similarity(input_vec, clean_vecs)[0]
    top = sims.argsort()[-10:][::-1]
    candidates = [clean_texts[i] for i in top]
    random.shuffle(candidates)
    return candidates[:n]

# Train model from feedback data only
@st.cache_resource(show_spinner="🧠 Training model...")
def train_model():
    df = load_feedback_data()
    if df.empty:
        st.warning("⚠ No training data available. Please give some input first.")
        return None, 0.0

    # Ensure all classes exist at least once
    for lbl in [0, 1, 2]:
        if lbl not in df["class"].unique():
            df = pd.concat([df, pd.DataFrame({
                "tweet": [f"dummy comment for label {lbl}"],
                "class": [lbl]
            })], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["tweet"], df["class"], test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/hate_offensive_model.joblib")
    return model, acc

# ------------------ Streamlit UI -------------------
st.set_page_config(page_title="Comment Cleanser", page_icon="✨")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("✨ Comment Cleanser")
st.markdown("<h2 style='font-size: 24px;'>Hate & Offensive Comment Detector</h2>", unsafe_allow_html=True)
st.markdown("This app detects hate/offensive content and gives polite suggestions.", unsafe_allow_html=True)

model, accuracy = train_model()
if model:
    st.markdown(f"📊 **Model Accuracy:** `{accuracy:.2%}`")

    user_input = st.text_area("✍️ Enter a tweet/comment to check:", height=120)

    if st.button("Check Comment Type") and user_input.strip():
        prediction = model.predict([user_input])[0]
        prob = model.predict_proba([user_input])[0]

        if prediction >= len(prob):
            st.error("⚠️ Model error: output class mismatch.")
        else:
            max_prob = prob[prediction]
            threshold = 0.65

            if max_prob < threshold:
                st.warning("🤔 This comment seems new or unclear to the model.")
                save_unknown_comment(user_input, prob)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("🧠 Saved. Re-training with feedback...")
                st.rerun()
            else:
                st.success(f"🔎 **Prediction:** {label_map[prediction]}")
                st.info(f"📊 Confidence:\n- Hate: `{prob[0]:.2%}`\n- Offensive: `{prob[1]:.2%}`\n- Clean: `{prob[2]:.2%}`")
                if prediction in [0, 1]:
                    st.markdown("💡 **Suggested Clean Alternatives:**")
                    for s in get_clean_suggestions(user_input, model):
                        st.write(f"• _{s}_")
else:
    st.stop()
