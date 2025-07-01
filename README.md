# 📱 SMS Spam Detector

A simple and interactive SMS Spam Detection web app built using **Python**, **scikit-learn**, and **Streamlit**. This app classifies SMS messages as either **Spam** or **Ham (not spam)** using a Naive Bayes model trained on the **SMS Spam Collection Dataset**.

---

## 🔍 Features

- Real-time classification of SMS messages
- Probability scores for spam and ham
- Displays model accuracy and classification report
- Simple and user-friendly web interface

---

## 🛠 How to Run Locally

1. **Clone the repository:**

```bash
git clone https://github.com/rafaelm200314/spamcsvmodel.git
cd spamcsvmodel

2. **Install the dependencies:**

pip install -r requirements.txt

3. **Run the Streamlit app:**
streamlit run app.py
Then open your browser and go to http://localhost:8501

🌐 Live Demo
https://rafaelm200314-spamcsvmodel-app-m3lsyb.streamlit.app/

📸 Screenshot
img/screenshot1.png
📊 Dataset

This app uses the SMS Spam Collection Dataset, which contains 5,574 labeled SMS messages (spam or ham).

⸻

🧠 Model Details
	•	Vectorizer: CountVectorizer (Bag-of-Words)
	•	Classifier: Multinomial Naive Bayes
	•	Accuracy: ~98% on test set (may vary slightly per run)

⸻

🚀 Future Improvements
	•	Add text preprocessing (stopword removal, stemming)
	•	Save and load trained model/vectorizer to reduce startup time
	•	Use TfidfVectorizer for better term weighting
	•	Improve UI layout and mobile responsiveness
```
