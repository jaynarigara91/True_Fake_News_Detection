import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import pickle
from bs4 import BeautifulSoup
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the pre-trained LSTM model
model = load_model(r'C:\Users\ADMIN\OneDrive\Desktop\Git\True_Fake_News_Sentiment\True_Fake_News_Sentiment\lstm_model.h5')

# Load the saved Tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Set the maximum sequence length (should match what was used during training)
maxlen = 300

# Function to clean and preprocess the input text
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    final_text = []
    for word in text.split():
        if word.strip().lower() not in stop:
            final_text.append(word.strip())
    cleaned_text = " ".join(final_text)
    
    # Check if the cleaned text is empty
    if not cleaned_text:
        cleaned_text = "placeholder"  # Add a placeholder word if empty

    return cleaned_text

# Function to preprocess and tokenize the input text
def preprocess_text(text):
    cleaned = clean_text(text)  # Clean the input text
    sequences = tokenizer.texts_to_sequences([cleaned])
    padded = sequence.pad_sequences(sequences, maxlen=maxlen)
    return padded

# Streamlit App
st.title("Fake News Detection with LSTM")

# Input from the user
user_input = st.text_area("Enter the news text:", height=150)

# Predict button
if st.button("Check News Authenticity"):
    if user_input:
        # Preprocess the input text
        processed_input = preprocess_text(user_input)
        
        # Make predictions
        prediction = model.predict(processed_input)
        prediction_label = "Real" if prediction > 0.5 else "Fake"
        
        # Display the result
        st.success(f"The News is {prediction_label}")
    else:
        st.warning("Please enter some text to analyze.")

# Display tips for better input
# st.info("Tip: The more detailed the text, the better the accuracy of the prediction.")

# Optional: Add footer or other information
st.markdown("---")
st.markdown("Developed by [Jay Narigara](https://www.linkedin.com/in/jaynarigara/)")
