import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle as pkl
import nltk
from nltk.corpus import stopwords

# Download stopwords quietly
nltk.download('stopwords', quiet=True)

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="✈️")

# Preprocessing function (same as training)
def preprocess_text(text):
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pkl.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Prediction function
def predict_sentiment(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    prediction = model.predict(padded)
    labels = ['negative', 'neutral', 'positive']
    return labels[prediction.argmax()]


# UI
st.markdown("""
<style>

/* background */
[data-testid="stAppViewContainer"] {
    background-color: #000000;
}

/* Main text color */
h1, h2, h3, p, label {
    color: white !important;
}

/* Input box */
textarea {
    background-color: #1e1e1e !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #555 !important;
}

/* Button */
button {
    background-color: #4CAF50 !important;
    color: white !important;
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)

# Title/description
st.title("✈️ Airline Sentiment Analyzer")
st.markdown("---")
st.markdown(
    "<p style='text-align: right; color: gray;'>Built by Sheema</p>",
    unsafe_allow_html=True
)
st.markdown("**Enter a sentence and check its sentiment.**")

# Input box
user_input = st.text_area("Type your sentence here:")

# Button
if st.button("Predict"):
    
    if user_input.strip() != "":
        result = predict_sentiment(user_input)

        if result == "positive":
            st.success(f"Sentiment: 😊{result}")

        elif result == "negative":
            st.error(f"Sentiment: 😠{result}")

        else:
            st.warning(f"Sentiment: 😐{result}")

    else:
        st.warning("Please enter some text!")