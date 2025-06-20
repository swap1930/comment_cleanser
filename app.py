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

# Labels
label_map = {0: "Hate Speech 😠", 1: "Offensive 😡", 2: "Clean 😊"}

# Save unknown low-confidence inputs
def save_unknown_comment(text, probs):
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/new_data.csv", mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([text, probs[0], probs[1], probs[2]])

# Load hate/offensive dataset with feedback
@st.cache_data(show_spinner="📥 Loading dataset...")
def load_data():
    df = pd.read_csv("data/train.csv", usecols=["tweet", "class"]).dropna()
    df["class"] = df["class"].astype(int)

    if os.path.exists("feedback/new_data.csv"):
        new_df = pd.read_csv("feedback/new_data.csv", header=None,
                             names=["tweet", "hate", "offensive", "clean"])
        new_df["class"] = new_df[["hate", "offensive", "clean"]].astype(float).idxmax(axis=1).map({
            "hate": 0, "offensive": 1, "clean": 2
        })
        df = pd.concat([df, new_df[["tweet", "class"]]], ignore_index=True)

    return df

# Load positive clean dataset
@st.cache_data
def load_positive_data():
    import re
    pos_df = pd.read_csv("data/positive_data.csv").dropna()

    # Remove trailing [x] tags
    def clean_text(text):
        return re.sub(r"\s*\[\d+\]$", "", text).strip()

    pos_df["text"] = pos_df["text"].apply(lambda x: clean_text(str(x)))
    pos_df = pos_df[pos_df["text"].apply(lambda x: isinstance(x, str) and len(x.split()) > 4)]
    pos_df = pos_df.drop_duplicates(subset="text").reset_index(drop=True)
    return pos_df["text"].tolist()


def get_clean_suggestions(user_input, model, n=3):
    positive_texts = load_positive_data()

    if not positive_texts:
        return ["No positive suggestions available."]

    vectorizer = model.named_steps["tfidf"]
    input_vec = vectorizer.transform([user_input])
    positive_vecs = vectorizer.transform(positive_texts)

    similarities = cosine_similarity(input_vec, positive_vecs)[0]
    top_indices = similarities.argsort()[-10:][::-1]  # pick top 10 similar

    # Shuffle to avoid repetition of top 3 same results
    top_candidates = [positive_texts[i] for i in top_indices]
    random.shuffle(top_candidates)

    return top_candidates[:n]

# Train the model
@st.cache_resource(show_spinner="🧠 Training model...")
def train_model():
    df = load_data()  # मुख्य data (feedback सहित) load कर

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

# UI
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.set_page_config(page_title="Comment Cleanser", page_icon="✨")


st.title(" ✨ Comment Cleanser")
st.markdown("<h1 style='font-size: 30px;'>Hate & Offensive Comment Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 19px; margin-bottom:30px'>Comment Cleanser detects offensive or harmful language in comments using a smart ML model. It then suggests polite and respectful alternatives to encourage positive communication.</p>", unsafe_allow_html=True)

model, accuracy = train_model()
st.markdown(f"📊 **Model Accuracy:** `{accuracy:.2%}`")


user_input = st.text_area("✍️ Enter a tweet/comment to check:", height=120)

if st.button("Check Comment Type"):
    if user_input.strip():  
        prediction = model.predict([user_input])[0]
        prob = model.predict_proba([user_input])[0]

        max_prob = prob[prediction]
        threshold = 0.65

        if max_prob < threshold:
            st.warning("🤔 This comment seems unclear to the model.")
            save_unknown_comment(user_input, prob)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("🧠 Saved & Re-training with feedback...")
            st.rerun()
        else:
            st.success(f"🔎 **Prediction:** {label_map[prediction]}")
            st.info(f"📊 Confidence Scores:\n- Hate: `{prob[0]:.2%}`\n- Offensive: `{prob[1]:.2%}`\n- Clean: `{prob[2]:.2%}`")

            if prediction in [0, 1]:  # Only suggest clean responses for bad input
                suggestions = get_clean_suggestions(user_input, model)
                st.markdown("💡 **Suggested Clean Alternatives:**")
                for s in suggestions:
                    st.write(f"• _{s}_")
    else:
        st.warning("⚠️ Please enter a comment.")
