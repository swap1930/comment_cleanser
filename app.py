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

# Label mapping
label_map = {0: "Hate Speech 😠", 1: "Offensive 😡", 2: "Clean 😊"}

# Save low-confidence comment to feedback
def save_unknown_comment(text, probs):
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/new_data.csv", mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([text, probs[0], probs[1], probs[2]])

# Load main dataset (once)
@st.cache_data(show_spinner="📥 Loading main dataset...")
def load_main_data():
    df = pd.read_csv("data/train.csv", usecols=["tweet", "class"]).dropna()
    df["class"] = df["class"].astype(int)
    return df

# Load positive comments
@st.cache_data
def load_positive_data():
    pos_df = pd.read_csv("data/positive_data.csv").dropna()
    pos_df["text"] = pos_df["text"].apply(lambda x: re.sub(r"\s*\[\d+\]$", "", str(x)).strip())
    pos_df = pos_df[pos_df["text"].apply(lambda x: isinstance(x, str) and len(x.split()) > 4)]
    pos_df = pos_df.drop_duplicates(subset="text").reset_index(drop=True)
    return pos_df["text"].tolist()

# Suggest clean responses based on similarity
def get_clean_suggestions(user_input, model, n=3):
    texts = load_positive_data()
    if not texts:
        return ["No suggestions available."]
    vec = model.named_steps["tfidf"]
    input_vec = vec.transform([user_input])
    text_vecs = vec.transform(texts)
    similarities = cosine_similarity(input_vec, text_vecs)[0]
    top = similarities.argsort()[-10:][::-1]
    random.shuffle(top)
    return [texts[i] for i in top[:n]]

# Train model on main dataset (cached)
@st.cache_resource(show_spinner="🧠 Training base model...")
def train_main_model():
    df = load_main_data()
    X_train, X_test, y_train, y_test = train_test_split(df["tweet"], df["class"], test_size=0.2, random_state=42)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    joblib.dump(model, "model/base_model.joblib")
    return model, acc

# Retrain only on feedback data
def retrain_on_feedback():
    if not os.path.exists("feedback/new_data.csv"):
        return None
    try:
        df = pd.read_csv("feedback/new_data.csv", header=None, names=["tweet", "hate", "offensive", "clean"])
        df["class"] = df[["hate", "offensive", "clean"]].astype(float).idxmax(axis=1).map({
            "hate": 0, "offensive": 1, "clean": 2
        })
        X = df["tweet"]
        y = df["class"]
        model = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
        ])
        model.fit(X, y)
        return model
    except Exception as e:
        print("Retrain error:", e)
        return None

# UI Setup
st.set_page_config(page_title="Comment Cleanser", page_icon="✨")
st.title(" ✨ Comment Cleanser")
st.markdown("<h4>Detect Hate, Offensive & Clean Comments</h4>", unsafe_allow_html=True)

# Load main model
base_model, accuracy = train_main_model()
st.markdown(f"📊 **Base Model Accuracy:** `{accuracy:.2%}`")

user_input = st.text_area("✍️ Enter a comment to check:", height=120)

# On prediction
if st.button("Check Comment Type"):
    if user_input.strip():
        prediction = base_model.predict([user_input])[0]
        prob = base_model.predict_proba([user_input])[0]
        max_prob = prob[prediction]
        threshold = 0.65

        if max_prob < threshold:
            st.warning("🤔 Unclear comment. Storing for future learning...")
            save_unknown_comment(user_input, prob)

            # Try retraining on new feedback data
            feedback_model = retrain_on_feedback()
            if feedback_model:
                st.success("🔁 Retrained on feedback!")
                prediction = feedback_model.predict([user_input])[0]
                prob = feedback_model.predict_proba([user_input])[0]
            else:
                st.error("⚠️ Could not retrain on feedback.")
                feedback_model = base_model
        else:
            feedback_model = base_model

        st.success(f"🔎 **Prediction:** {label_map[prediction]}")

        # Safely show confidence scores
        classes = feedback_model.named_steps['clf'].classes_
        confidence_msg = "📊 Confidence:\n"
        if 0 in classes:
            confidence_msg += f"- Hate: `{prob[list(classes).index(0)]:.2%}`\n"
        if 1 in classes:
            confidence_msg += f"- Offensive: `{prob[list(classes).index(1)]:.2%}`\n"
        if 2 in classes:
            confidence_msg += f"- Clean: `{prob[list(classes).index(2)]:.2%}`\n"
        st.info(confidence_msg)

        # Clean suggestions if offensive/hate
        if prediction in [0, 1]:
            st.markdown("💡 **Suggested Clean Alternatives:**")
            for s in get_clean_suggestions(user_input, feedback_model):
                st.write(f"• _{s}_")
    else:
        st.warning("⚠️ Please enter a comment.")
