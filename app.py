import pandas as pd
data=pd.read_csv("Tweets.csv")
data.head()

import nltk # NLP
from nltk.corpus import stopwords # stopwords 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 

# Download stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english')) # get stopwords

# preprocessing function
def preprocess_text(text): 
    text = text.lower()   # Converts text to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words]) 
    return text

# Apply preprocessing to the text column
data['text'] = data['text'].apply(preprocess_text)

data["text"]

# Tokenization and padding 
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
# sequences

padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
padded_sequences

data["airline_sentiment"].value_counts()

# Extract labels
labels = pd.get_dummies(data['airline_sentiment']).values
labels #pos,neg,neutral

# training the model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# Building the RNN model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Train the model
num_epochs = 2
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Function to preprocess and predict sentiment
def predict_sentiment(text, tokenizer, model):
    # Preprocess the text (tokenization and padding)
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    print(f"The padded seq is {padded_sequence}")

    # Make a prediction
    prediction = model.predict(padded_sequence)
    print(f"The prediction is {prediction}")

    # Get the sentiment with the highest probability
    sentiment_index = prediction.argmax(axis=1)[0]
    print(f" The sentiment index is {sentiment_index}")

    # Map index to sentiment label
    sentiment_labels = ['negative', 'neutral', 'positive']
    sentiment = sentiment_labels[sentiment_index]

    return sentiment

# Example usage
new_text = "I'm really happy with the service provided by the airline!"
predicted_sentiment = predict_sentiment(new_text, tokenizer, model)
print(f"The predicted sentiment is: {predicted_sentiment}")

# Save model
model.save("model.h5")

#save tokenizer
import pickle as pkl
with open("tokenizer.pkl", "wb") as f:
    pkl.dump(tokenizer, f)