# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analysis", layout='centered')
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.markdown('Enter a movie review below to classify it as **Positive** or **Negative**.')

# Example input
with st.expander("🔍 Example Review"):
    st.markdown("> *The movie was absolutely fantastic with great acting and plot!*")

# Input area
user_input = st.text_area('✍️ Enter your review here:')

# Classification
if st.button('Classify Review'):
    if user_input.strip() == '':
        st.warning('⚠️ Please enter a review before classifying.')
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)[0][0]
        sentiment = 'Positive 😊' if prediction > 0.5 else 'Negative 😞'

        # Output
        st.markdown(f"### Sentiment: **{sentiment}**")
        st.progress(int(prediction * 100) if prediction > 0.5 else int((1 - prediction) * 100))
        st.markdown(f"**Confidence Score:** `{prediction:.4f}`")

else:
    st.info('🧠 Enter a review and click the button to classify it.')

