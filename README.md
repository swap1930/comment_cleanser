# 🧹 Comment Cleanser – Hate & Offensive Comment Detector

A Streamlit-powered web app that detects whether a tweet/comment is **Hate Speech**, **Offensive**, or **Clean** using Machine Learning. It also learns from uncertain inputs and suggests cleaner alternatives.

---

## 🚀 Features

- Detects:  
  ✅ Hate Speech  
  ✅ Offensive Language  
  ✅ Clean Comments

- Self-Learning:  
  Saves **new, unclear comments** and re-trains the model with feedback.

- Suggestions:  
  Gives **clean alternative comments** similar to offensive ones.

---

## 📁 Project Structure

📦 comment-cleanser/
├── app.py # Main Streamlit app
├── data/
│ └── train.csv # Dataset with "tweet" and "class" columns
├── feedback/
│ └── new_data.csv # Saved unclear comments (auto-created)
├── model/
│ └── hate_offensive_model.joblib # Trained model (auto-created)
└── requirements.txt # Python dependencies

yaml
Copy
Edit

---

## 📦 Setup Instructions

1. **Clone this repo:**

```bash
git clone https://github.com/your-username/comment-cleanser.git
cd comment-cleanser
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py
🧠 Dataset Used
Dataset: Hate Speech and Offensive Language Dataset

Labels:

0 – Hate Speech 😠

1 – Offensive 😡

2 – Clean 😊

💡 Future Ideas
Add support for user feedback correction

Deploy as a web service or API

Use deep learning for more accuracy

🧑‍💻 Author
Swapnil Dudhane
SYIT, GPKP
🌐 GitHub: your-username
