import sys
import pandas as pd
import numpy as np


# NLP Package
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import words
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.utils import to_categorical

print("Version: ", tf.__version__)  # Check tf version
print("GPU is",
      "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")  # Check GPU status
physical_devices = tf.config.experimental.list_physical_devices('GPU')  # Config GPU
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Misc.
import re

OUTPUT_PATH = "data/labelled.csv"


def main(input_path):
    # read data here
    df = pd.read_csv(input_path)

    # IMPLEMENT DATA TRANSFORMATION HERE (IF REQUIRED)
    df.drop(labels='id', axis=1, inplace=True)

    # Functions for text preprocessing
    def lowercase(text):
        return [word.lower() for word in text]

    def remove_non_letters(text):
        return [re.sub('[^a-z\s]', '', word) for word in text]

    def tokenizer(text):
        return text.split(' ')

    lemmatizer = WordNetLemmatizer()

    def lemmatization(text):
        # Lemmatization (it's almost always better than stemming...)
        return [lemmatizer.lemmatize(word) for word in text]

    nltk_stopwords = stopwords.words("english")
    spacy_stopwords = list(STOP_WORDS)
    final_stopwords = list(set(nltk_stopwords + spacy_stopwords))

    def remove_stopword(text):
        return [word for word in text if word not in final_stopwords]

    def remove_whitespace(text):
        return [word for word in text if word != '']

    def remove_handle(text):
        return [word for word in text if '@' not in word]

    def convert_to_string(text):
        return ' '.join(text)

    words_list = words.words()

    def filter_gibberish(text):
        return [word for word in text if word in words_list]

    def data_transform(content_col):
        df[content_col] = df[content_col].apply(tokenizer)
        df[content_col] = df[content_col].apply(lowercase)
        df[content_col] = df[content_col].apply(remove_handle)
        df[content_col] = df[content_col].apply(remove_non_letters)
        df[content_col] = df[content_col].apply(remove_stopword)
        df[content_col] = df[content_col].apply(lemmatization)
        df[content_col] = df[content_col].apply(remove_stopword)
        df[content_col] = df[content_col].apply(remove_whitespace)
        df[content_col] = df[content_col].apply(filter_gibberish)
        df[content_col] = df[content_col].apply(convert_to_string)
        return content_col

    # Encoding Sentiments
    def sentiments_encoding(sentiment):
        sentiments = ['happy', 'sad', 'neutral', 'fury']
        return sentiments.index(sentiment)

    df['sentiment'] = df['sentiment'].apply(sentiments_encoding)

    df['content'] = data_transform('content')

    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 250

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['content'].values)
    X = tokenizer.texts_to_sequences(df['content'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

    y = df['sentiment']
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # IMPLEMENT MODEL LOADING HERE
    def create_model():
        embedding_dim = 50

        model = Sequential()
        model.add(Embedding(input_dim=MAX_NB_WORDS,
                            output_dim=embedding_dim,
                            input_length=MAX_SEQUENCE_LENGTH))
        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(layers.Dropout(0.2))

        model.add(Dense(512, activation='relu'))
        model.add(layers.Dropout(0.2))

        model.add(Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))

        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
    model = create_model().fit(X_train, y_train,
                    epochs=50,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    batch_size= 64)

    # IMPLEMENT MODEL INFERENCE HERE
    new_model = tf.keras.models.load_model(model)
    new_model.summary()
    
    predictions = new_model.predict(df)
    prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv')

    # ------ DUMMY TRANSFORMATION -------

    possible_predictions = ["happy", "sad", "neutral", "fury"]
    np.random.seed = 1
    dummy_predictions = np.random.choice(a=possible_predictions, size=len(df))
    dummy_predictions_df = pd.DataFrame(dummy_predictions)

    # --- END OF DUMMY TRANSFORMATION ---

    # SAVE PREDICTIONS TO CSV
    dummy_predictions_df.to_csv(OUTPUT_PATH, index=False, header=False)


main(r"C:\Users\Darryl See\Desktop\NTUCampCode x Roboto Hackathon\challenge1\data\train.csv")