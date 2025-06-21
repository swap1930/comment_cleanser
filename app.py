import streamlit as st
import pandas as pd
import os
import csv
import re
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Labels
label_map = {0: "Hate Speech 😠", 1: "Offensive 😡", 2: "Clean 😊"}

# Save unknown inputs (low confidence)
def save_unknown_comment(text, probs):
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/new_data.csv", mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([text, probs[0], probs[1], probs[2]])

# First time training on train.csv only
@st.cache_resource
def initial_train_model():
    if os.path.exists("model/hate_offensive_model.joblib"):
        return joblib.load("model/hate_offensive_model.joblib"), 1.0  # load dummy acc
    df = pd.read_csv("data/train.csv", usecols=["tweet", "class"]).dropna()
    df["class"] = df["class"].astype(int)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])
    X_train, X_test, y_train, y_test = train_test_split(df["tweet"], df["class"], test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/hate_offensive_model.joblib")
    return model, model.score(X_test, y_test)

# Only retrain on feedback data
def retrain_on_feedback():
    if not os.path.exists("feedback/new_data.csv"):
        return
    df = pd.read_csv("feedback/new_data.csv", header=None, names=["tweet", "hate", "offensive", "clean"])
    df["class"] = df[["hate", "offensive", "clean"]].astype(float).idxmax(axis=1).map({
        "hate": 0, "offensive": 1, "clean": 2
    })
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])
    model.fit(df["tweet"], df["class"])
    joblib.dump(model, "model/hate_offensive_model.joblib")

# Load clean suggestion dataset
@st.cache_data
def load_positive_data():
    pos_df = pd.read_csv("data/positive_data.csv").dropna()
    def clean_text(text): return re.sub(r"\s*\[\d+\]$", "", text).strip()
    pos_df["text"] = pos_df["text"].apply(lambda x: clean_text(str(x)))
    pos_df = pos_df[pos_df["text"].apply(lambda x: isinstance(x, str) and len(x.split()) > 4)]
    pos_df = pos_df.drop_duplicates(subset="text").reset_index(drop=True)
    return pos_df["text"].tolist()

# Suggest positive alternatives
def get_clean_suggestions(user_input, model, n=3):
    texts = load_positive_data()
    if not texts: return ["No suggestions found."]
    vectorizer = model.named_steps["tfidf"]
    input_vec = vectorizer.transform([user_input])
    text_vecs = vectorizer.transform(texts)
    sims = cosine_similarity(input_vec, text_vecs)[0]
    top_idxs = sims.argsort()[-10:][::-1]
    top_texts = [texts[i] for i in top_idxs]
    random.shuffle(top_texts)
    return top_texts[:n]

# --- Streamlit UI ---
st.set_page_config(page_title="Comment Cleanser", page_icon="🧹")
st.title("🧹 Comment Cleanser – Hate & Offensive Comment Detector")
st.markdown("<small>Built with Logistic Regression + TF-IDF</small>", unsafe_allow_html=True)

model, acc = initial_train_model()
st.markdown(f"**Model Accuracy:** `{acc:.2%}`")

user_input = st.text_area("✍️ Enter a tweet/comment to check:")

if st.button("Check Comment Type"):
    if user_input.strip():
        prediction = model.predict([user_input])[0]
        prob = model.predict_proba([user_input])[0]
        max_prob = prob[prediction]
        threshold = 0.65

        if max_prob < threshold:
            st.warning("🤔 Comment unclear to model. Learning from it.")
            save_unknown_comment(user_input, prob)
            retrain_on_feedback()
            st.success("✅ Feedback saved & model retrained.")
        else:
            st.success(f"🔎 Prediction: **{label_map[prediction]}**")
            st.info(f"📊 Confidence: Hate: `{prob[0]:.2%}`, Offensive: `{prob[1]:.2%}`, Clean: `{prob[2]:.2%}`")
            if prediction in [0, 1]:
                st.markdown("💡 **Suggested Clean Alternatives:**")
                for suggestion in get_clean_suggestions(user_input, model):
                    st.write(f"• _{suggestion}_")
    else:
        st.warning("⚠️ Please enter a comment.")
