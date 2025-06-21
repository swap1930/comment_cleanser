import streamlit as st
import pandas as pd
import os
import csv
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Mapping for labels
label_map = {0: "Hate Speech 😠", 1: "Offensive 😡", 2: "Clean 😊"}

# Save unknown/low confidence inputs to feedback
def save_unknown_comment(text, probs):
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/new_data.csv", mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([text, probs[0], probs[1], probs[2]])

# Load only feedback data
@st.cache_data(show_spinner="📥 Loading feedback data...")
def load_feedback_data():
    if os.path.exists("feedback/new_data.csv"):
        df = pd.read_csv("feedback/new_data.csv", header=None,
                         names=["tweet", "hate", "offensive", "clean"])
        df["class"] = df[["hate", "offensive", "clean"]].astype(float).idxmax(axis=1).map({
            "hate": 0, "offensive": 1, "clean": 2
        })
        return df[["tweet", "class"]]
    else:
        return pd.DataFrame(columns=["tweet", "class"])

# Load clean comments dataset for suggestions
@st.cache_data
def load_positive_data():
    pos_df = pd.read_csv("data/positive_data.csv").dropna()
    def clean_text(text): return re.sub(r"\s*\[\d+\]$", "", text).strip()
    pos_df["text"] = pos_df["text"].apply(lambda x: clean_text(str(x)))
    pos_df = pos_df[pos_df["text"].apply(lambda x: len(x.split()) > 4)].drop_duplicates(subset="text")
    return pos_df["text"].tolist()

# Suggest positive clean alternatives
def get_clean_suggestions(user_input, model, n=3):
    positive_texts = load_positive_data()
    if not positive_texts:
        return ["No positive suggestions available."]
    vectorizer = model.named_steps["tfidf"]
    input_vec = vectorizer.transform([user_input])
    positive_vecs = vectorizer.transform(positive_texts)
    similarities = cosine_similarity(input_vec, positive_vecs)[0]
    top_indices = similarities.argsort()[-10:][::-1]
    top_candidates = [positive_texts[i] for i in top_indices]
    random.shuffle(top_candidates)
    return top_candidates[:n]

# Train only on feedback data
@st.cache_resource(show_spinner="🧠 Training on feedback data only...")
def train_model():
    df = load_feedback_data()
    if df.empty:
        return None, 0.0
    X_train, X_test, y_train, y_test = train_test_split(df["tweet"], df["class"], test_size=0.2, random_state=42)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/hate_offensive_model.joblib")
    return model, acc

# Hide Streamlit's default UI elements
st.markdown("""
    <style>
        #MainMenu, header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(page_title="Comment Cleanser", page_icon="✨")
st.title("✨ Comment Cleanser")
st.markdown("<h3 style='font-size: 24px;'>Feedback-based Hate & Offensive Comment Detector</h3>", unsafe_allow_html=True)

model, accuracy = train_model()
if model is None:
    st.warning("⚠ No feedback data available. Please enter a comment to start training.")
    st.stop()

st.markdown(f"📊 **Model Accuracy (on feedback data):** `{accuracy:.2%}`")

user_input = st.text_area("✍️ Enter a tweet/comment to check:", height=120)

if st.button("Check Comment Type"):
    if user_input.strip():
        prediction = model.predict([user_input])[0]
        prob = model.predict_proba([user_input])[0]
        max_prob = prob[prediction]
        threshold = 0.65

        if max_prob < threshold:
            st.warning("🤔 Model is unsure about this comment. Saving it for learning...")
            save_unknown_comment(user_input, prob)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("🧠 Saved & retraining...")
            st.rerun()
        else:
            st.success(f"🔎 **Prediction:** {label_map[prediction]}")
            st.info(f"📊 Confidence Scores:\n- Hate: `{prob[0]:.2%}`\n- Offensive: `{prob[1]:.2%}`\n- Clean: `{prob[2]:.2%}`")
            if prediction in [0, 1]:
                suggestions = get_clean_suggestions(user_input, model)
                st.markdown("💡 **Suggested Clean Alternatives:**")
                for s in suggestions:
                    st.write(f"• _{s}_")
    else:
        st.warning("⚠️ Please enter a comment.")
