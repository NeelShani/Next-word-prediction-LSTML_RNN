import streamlit as st 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle

##load the LSTM
model = load_model('next_word_lstm.h5')
model_GRU = load_model('next_word_gru.h5')

##load the tkenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer= pickle.load(handle)
    

#the prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list =token_list[-(max_sequence_len-1):]
    token_list=pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

## streamlit 

st.title("Predict the next word")

st.header("Simple LSTM RNN")


input_text = st.text_input('Enter the sequence of word', 'I only follow the path shown by')
if st.button('Predict'):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')
    

st.header("GRU LSTM Model")

input_text = st.text_input('Enter the sequence of word', 'I only follow the path shown by ')
if st.button('Predict '):
    max_sequence_len = model_GRU.input_shape[1]+1
    next_word = predict_next_word(model_GRU, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')