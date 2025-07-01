import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare data
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['text'] = df['text'].str.lower()

# Split and vectorize
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Streamlit app
st.title("📱 SMS Spam Classifier")
st.write("Enter an SMS message below to check if it's spam or not:")

user_input = st.text_area("Your Message", height=100)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # Preprocess and predict
        user_input_lower = user_input.lower()
        user_input_vec = vectorizer.transform([user_input_lower])
        prediction = model.predict(user_input_vec)[0]
        prediction_proba = model.predict_proba(user_input_vec)[0]

        st.subheader("🔍 Prediction:")
        st.success(f"The message is **{prediction.upper()}**.")

        st.subheader("📊 Probability:")
        st.write(f"Ham: {prediction_proba[0]:.4f}, Spam: {prediction_proba[1]:.4f}")

st.sidebar.title("📈 Model Performance")
if st.sidebar.checkbox("Show Accuracy and Report"):
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    st.sidebar.markdown(f"**Accuracy:** {accuracy:.4f}")
    st.sidebar.text("Classification Report:")
    st.sidebar.text(report)