# 🧹 Comment Cleanser – Hate & Offensive Comment Detector

This is a Streamlit-based machine learning web app that detects whether a user-provided comment or tweet is categorized as **Hate Speech**, **Offensive**, or **Clean**. The model is trained using logistic regression and can **improve over time** by learning from newly encountered comments.

---

## 📌 Features

- 🔍 Classifies input into:
  - Hate Speech 😠
  - Offensive 😡
  - Clean 😊
- 📥 Learns from unknown/uncertain inputs (low confidence)
- 💡 Suggests clean alternative comments when toxic input is detected
- 🧠 Re-trains model using both original and feedback data
- 💬 Built with Streamlit for quick deployment and interaction

---

## 📂 Folder Structure

comment-cleanser/
├── app.py # Main Streamlit app
├── data/
│ └── train.csv # Original dataset (tweet + class)
├── feedback/
│ └── new_data.csv # New user comments (auto-generated)
├── model/
│ └── hate_offensive_model.joblib # Trained model (auto-saved)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

## 💾 Dataset

We use the [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/lisaleesmith/hate-speech-and-offensive-language-dataset).

- `tweet` — Comment or tweet text
- `class` — Label:
  - 0 = Hate Speech
  - 1 = Offensive
  - 2 = Clean

---

## 🚀 Getting Started

### 🔧 Installation

```bash
# Clone this repository
git clone https://github.com/your-username/comment-cleanser.git
cd comment-cleanser

# Create and activate a virtual environment
python -m venv env
env\Scripts\activate  # Windows
# OR
source env/bin/activate  # macOS/Linux

# Install the dependencies
pip install -r requirements.txt
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
📊 How It Works
User submits a comment.

The model classifies the comment into one of three categories.

If prediction confidence is low (<65%), it stores the comment for learning.

Suggestions are shown if the comment is toxic.

App automatically re-trains using new data on-the-fly.

📸 Example Output
Input: You're a dumb person

Prediction: Offensive 😡

Confidence: 91%

Suggested Clean Comments:

You're really smart!

That was a great point.

✍️ Author
Swapnil Dudhane
SYIT – Government Polytechnic, Kolhapur
🔗 GitHub

📜 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

### ✅ Notes:
- Change `your-username` to your actual GitHub username before uploading.
- Add a `LICENSE` file if you want GitHub to show the license properly.
- Let me know if you also want a demo GIF, app screenshot, or deployment guide.

Ready to paste. You're good to go now.








