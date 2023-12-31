# -*- coding: utf-8 -*-
"""Submission 1 - Machine Learning Terapan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ueBtrZlFDBqbtC8QXiypHA1QeB1HH10v

# Import Library & Load File
"""

import re, string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS

from keras.preprocessing import text
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Menghubungkan notebook dengan Google Drive

from google.colab import drive
drive.mount('/content/drive/', force_remount = True)

true = pd.read_csv('/content/drive/My Drive/Colab Notebooks/True.csv')
false = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Fake.csv')

true.head()

false.head()

true['category'] = 1
false['category'] = 0

df = pd.concat([true,false])

df.head()

"""# Explore Dataset"""

df.info()

# Eksplor banyaknya data dari masing masing subjek artikel

type_df = df['subject'].value_counts().sort_values(ascending=False).head(10)
type_df = pd.DataFrame(type_df)
type_df = type_df.reset_index()

# Plotting
sns.set()
plt.figure(figsize=(15, 7))
type_plt = sns.barplot(x='index', y='subject', data=type_df)

for bar in type_plt.patches:
  type_plt.annotate(format(bar.get_height()),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=9, xytext=(0, 8),
                   textcoords='offset points')

plt.xlabel("\n Subject")
plt.ylabel("Number of News Subject")
plt.title("Type of News Subject\n")
plt.show()

# Eksplor kata yang sering muncul dari seluruh dataset dengan word cloud

wc = WordCloud(background_color="black", max_words=1000,
               max_font_size=256,
               random_state=123, width=1000, height=1000)
wc.generate(' '.join(map(str, df['text'])))
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 5))
plt.bar('Fake News', len(false), color='red')
plt.bar('Real News', len(true), color='blue')
plt.title('Distribution of Fake News and Real News', size=15)
plt.xlabel('News Type', size=15)
plt.ylabel('Count of News Articles', size=15)

df.isna().sum()

"""# Data Cleaning and Preparation

## Menggabungkan semua data teks ke dalam satu kolom
"""

df['text'] = df['text'] + " " + df['title']
df = df.drop(columns=['title', 'subject', 'date'])

df.tail()

nltk.download('stopwords')

# Mengambil stopwords dalam bahasa inggris

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

# Remove stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

# Remove noisy text
def clean_text(text):
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('\\W', ' ', text)
    text = re.sub('\n', '', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('^ ', '', text)
    text = re.sub(' $', '', text)
    text = remove_stopwords(text)
    return text

# Get cleaned text
df['text'] = df['text'].apply(clean_text)

"""# Train Test Split"""

# Membagi data latih dan data uji dengan proposisi 70:30

X = df['text'].values
y = df['category'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

max_features = 10000
maxlen = 300

# Tokenizing setiap kata dalam teks dan mapping pada data latih

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)

sequence_train = tokenizer.texts_to_sequences(X_train)
sequence_test = tokenizer.texts_to_sequences(X_test)

padded_train = pad_sequences(sequence_train, maxlen=maxlen)
padded_test = pad_sequences(sequence_test, maxlen=maxlen)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_features, output_dim=128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(padded_train,
                    y_train,
                    batch_size=128,
                    epochs=10,
                    steps_per_epoch=80,
                    validation_data=(padded_test, y_test),
                    verbose='auto')

model.evaluate(padded_test, y_test)

# Menampilkan grafik log model

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(acc, color='blue')
plt.plot(val_acc, color='red')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(loss, color='blue')
plt.plot(val_loss, color='red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

pred = model.predict(padded_test)

y_pred = np.where(pred > 0.5, 1, 0)

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
plt.subplots(figsize=(5, 5))
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu",
            fmt='g', yticklabels=['Fake', 'Original'],
            xticklabels = ['Fake','Original'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

