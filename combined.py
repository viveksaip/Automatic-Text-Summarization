import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Attention Layer Implementation
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W_a = self.add_weight(name='W_a', shape=(input_shape[0][2], input_shape[0][2]), initializer='uniform', trainable=True)
        self.U_a = self.add_weight(name='U_a', shape=(input_shape[1][2], input_shape[0][2]), initializer='uniform', trainable=True)
        self.V_a = self.add_weight(name='V_a', shape=(input_shape[0][2], 1), initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        encoder_out_seq, decoder_out_seq = inputs

        # Energy calculation
        e_i = tf.nn.tanh(tf.tensordot(encoder_out_seq, self.W_a, axes=[2, 0]) + tf.expand_dims(tf.tensordot(decoder_out_seq, self.U_a, axes=[2, 0]), 1))
        e_i = tf.nn.softmax(tf.tensordot(e_i, self.V_a, axes=[2, 0]), axis=1)
        context_vector = tf.reduce_sum(encoder_out_seq * tf.expand_dims(e_i, -1), axis=1)

        return context_vector, e_i

# Data Preprocessing Function
def text_cleaner(text, num):
    contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", ...}  # Add other contractions as needed
    stop_words = set(stopwords.words('english'))
    new_string = text.lower()
    new_string = BeautifulSoup(new_string, "lxml").text
    new_string = re.sub(r'\([^)]*\)', '', new_string)
    new_string = ' '.join([contraction_mapping.get(t, t) for t in new_string.split() if t not in stop_words])
    return new_string.strip()

# Load and clean the dataset
data = pd.read_csv("Reviews.csv", nrows=100000)
data.drop_duplicates(subset=['Text'], inplace=True)
data.dropna(axis=0, inplace=True)

data['cleaned_text'] = data['Text'].apply(lambda x: text_cleaner(x, 0))
data['cleaned_summary'] = data['Summary'].apply(lambda x: text_cleaner(x, 1))
data['cleaned_summary'] = data['cleaned_summary'].apply(lambda x: 'sostok ' + x + ' eostok')

# Split the data
x_train, x_val, y_train, y_val = train_test_split(data['cleaned_text'].values, data['cleaned_summary'].values, test_size=0.1, random_state=0)

# Tokenization
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(x_train)
x_train_seq = pad_sequences(x_tokenizer.texts_to_sequences(x_train), maxlen=30, padding='post')
x_val_seq = pad_sequences(x_tokenizer.texts_to_sequences(x_val), maxlen=30, padding='post')

y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(y_train)
y_train_seq = pad_sequences(y_tokenizer.texts_to_sequences(y_train), maxlen=8, padding='post')
y_val_seq = pad_sequences(y_tokenizer.texts_to_sequences(y_val), maxlen=8, padding='post')

# Model Definition
def build_seq2seq_model(vocab_size_x, vocab_size_y):
    encoder_input = Input(shape=(30,))
    encoder_embedding = Embedding(input_dim=vocab_size_x, output_dim=256)(encoder_input)
    encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    encoder_output, state_h, state_c = encoder_lstm(encoder_embedding)

    decoder_input = Input(shape=(8,))
    decoder_embedding = Embedding(input_dim=vocab_size_y, output_dim=256)(decoder_input)
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

    attention_layer = AttentionLayer()
    context_vector, attention_weights = attention_layer([encoder_output, decoder_output])

    combined = Concatenate()([context_vector, decoder_output])
    output = Dense(vocab_size_y, activation='softmax')(combined)

    model = Model([encoder_input, decoder_input], output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and train the model
vocab_size_x = len(x_tokenizer.word_index) + 1
vocab_size_y = len(y_tokenizer.word_index) + 1
model = build_seq2seq_model(vocab_size_x, vocab_size_y)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit([x_train_seq, y_train_seq], np.expand_dims(y_train_seq, -1), 
          validation_data=([x_val_seq, y_val_seq], np.expand_dims(y_val_seq, -1)), 
          epochs=10, batch_size=64, callbacks=[early_stopping])

# Save the model
model.save('summarization_model.h5')

# Prediction Function
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    encoder_output, state_h, state_c = model.layers[1].predict(input_seq)

    # Generate an empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = y_tokenizer.word_index['sostok']

    # Sampling loop for a batch of sequences (simplified)
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = model.layers[2].predict([encoder_output, target_seq])
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = y_tokenizer.index_word[sampled_token_index]
        
        if sampled_char == 'eostok' or len(decoded_sentence.split()) >= 8:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_char
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
    return decoded_sentence.strip()

# Example usage
input_text = "Your input text goes here."  # Replace with actual text
cleaned_input = text_cleaner(input_text, 0)
input_seq = pad_sequences(x_tokenizer.texts_to_sequences([cleaned_input]), maxlen=30, padding='post')
summary = decode_sequence(input_seq)
print("Generated Summary:", summary)
