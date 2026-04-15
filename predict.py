from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle as pkl
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# preprocessing function (same as training)
def preprocess_text(text):
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load model
model = load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pkl.load(f)

# Prediction function
def predict_sentiment(text, tokenizer, model):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')

    prediction = model.predict(padded_sequence)
    sentiment_index = prediction.argmax(axis=1)[0]

    sentiment_labels = ['negative', 'neutral', 'positive']
    return sentiment_labels[sentiment_index]

# Test
print(predict_sentiment("Very good service", tokenizer, model))
