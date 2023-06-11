#!pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr -q
#!pip install --upgrade accelerate

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import re
import string
import nltk
from collections import Counter
from datasets import load_dataset
import jieba
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

dataset = load_dataset("seamew/THUCNews")
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words("chinese"))

df_train = pd.DataFrame(dataset["train"])
df_val = pd.DataFrame(dataset["validation"])
df_test = pd.DataFrame(dataset["test"])


def remove_noise(text):
    chinese_punctuation = ["，","。","！","？","；","：","「","」","『","』","【","】","（","）","《","》","〈","〉","·","“","”","‘","’","(",")"]
    segmentedText = list(jieba.cut(text))
    no_punctuation = [word for word in segmentedText if word not in chinese_punctuation]
    no_filler_words = [word for word in no_punctuation if word not in stop]
    filtered_words = [word for word in no_filler_words if not re.match(r'^\d+(?:[.,]\d+)?[%]?$|^[\d.,]+$', word)]
    return filtered_words

df_train["text"] = df_train["text"].map(remove_noise)
df_val["text"] = df_val["text"].map(remove_noise)
df_test["text"] = df_test["text"].map(remove_noise)


def counter_word(dataset):
    counter = Counter()

    for example in dataset:
        counter.update(example)

    return counter

counter = counter_word(df_train["text"])
num_unique_words = len(counter)

train_sentences = [example for example in df_train['text']]
train_labels = [example for example in df_train['label']]
val_sentences = [example for example in df_val['text']]
val_labels = [example for example in df_val['label']]
test_sentences = [example for example in df_test['text']]
test_labels = [example for example in df_test['label']]

tokenizer = Tokenizer(num_words=num_unique_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

max_length = max(len(seq) for seq in train_sequences)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])
def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])

# If same output, everything is right so far
# print(decode(train_sequences[10]))
# print(train_sentences[10])

model = keras.models.Sequential()
model.add(Embedding(num_unique_words, 32, input_length=max_length))
# dropout parameter of 0.1 specifies that a fraction of the input units (10%) will be randomly set to 0 during training to prevent overfitting.
model.add(LSTM(64, dropout=0.1))
"""
10 because there are 10 different labels 

activation="softmax" takes a vector of real-valued numbers as input and transforms them into a probability distribution over K different classes
    - Use softmax for multiclassification problems 
"""
model.add(Dense(10, activation="softmax"))
model.summary()

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)

train_labels_onehot = tf.keras.utils.to_categorical(train_labels_encoded)
val_labels_onehot = tf.keras.utils.to_categorical(val_labels_encoded)
test_labels_onehot = tf.keras.utils.to_categorical(test_labels_encoded)

# Use CategoricalCrossentropy for multiclassification problems
loss = keras.losses.CategoricalCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics=["accuracy"]
model.compile(loss=loss, optimizer=optim, metrics=metrics)
model.fit(train_padded, train_labels_onehot, epochs=2, validation_data=(val_padded, val_labels_onehot), verbose=2)


predictions = model.predict(test_padded)
predicted_labels = np.argmax(predictions, axis=1)
accuracy = accuracy_score(test_labels_encoded, predicted_labels)
print("Accuracy:", accuracy)

"""
Custom predictions

Given the training data, only topics related to the following can be correctly predicted
  Sports, Entertainment, Property, Education, Fashion, Current affairs, Game, Sociology, Technology, Economics
"""
input_text = "我要做一个有关马克思主义的游戏。它一定会很好玩哟！你想不想玩？"
label_code = dataset["train"].info.features["label"].names
processed_text = pad_sequences(tokenizer.texts_to_sequences([remove_noise(input_text)]), maxlen=max_length, padding='post')
predicted_label = np.argmax(model.predict(processed_text), axis=1)
print(label_code[predicted_label[0]])
