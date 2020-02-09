#!/usr/bin/env python3

import csv
import pickle
import re
import string
from pprint import pprint
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.utils import plot_model, to_categorical


def preprocess(sentence: str) -> str:
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = filter(
        lambda token: token not in stopwords.words('english'), tokens)
    return ' '.join(filtered_words)


def scale_scores(df: pd.DataFrame) -> pd.DataFrame:
    hist = {}
    for i in set(df['category']):
        hist[i] = 0

    for i in df.index:
        x = hist[df.iloc[i]['category']]
        hist[df.iloc[i]['category']] = max(hist[df.iloc[i]['category']],
                                           df.iloc[i]['score'])

    score = np.array(df['score'])

    for i in df.index:
        if hist[df.iloc[i]['category']] != 0:
            score[i] *= 7 / hist[df.iloc[i]['category']]

    df = pd.DataFrame({
        'name': np.array(df['name']),
        'about': np.array(df['about']),
        'category': np.array(df['category']),
        'ing': np.array(df['ing']),
        'score': score
    })

    return df


def new_model(ing: np.ndarray, about: np.ndarray, category: np.ndarray,
              score: np.ndarray) -> models.Model:
    inputA = layers.Input(shape=91, name='about_model')
    inputB = layers.Input(shape=56, name='ing_model')
    inputC = layers.Input(shape=19, name='cat_model')

    embedA = layers.Embedding(input_dim=91, output_dim=300,
                              input_length=91)(inputA)
    embedB = layers.Embedding(input_dim=56, output_dim=300,
                              input_length=56)(inputB)

    hidden1 = layers.Dense(200,
                           activation='relu',
                           kernel_initializer='he_normal')(embedA)
    hidden2 = layers.Dense(100,
                           activation='relu',
                           kernel_initializer='he_normal')(hidden1)
    hidden3 = layers.Dense(200,
                           activation='relu',
                           kernel_initializer='he_normal')(embedB)
    hidden4 = layers.Dense(100,
                           activation='relu',
                           kernel_initializer='he_normal')(hidden3)

    about_ing_concat = layers.concatenate([hidden2, hidden4], axis=1)
    hidden5 = layers.Dense(50,
                           activation='relu',
                           kernel_initializer='he_normal')(inputC)
    about_ing_concat_flat = layers.Flatten()(about_ing_concat)
    concat = layers.concatenate([about_ing_concat_flat, hidden5], axis=1)
    final = layers.Dense(1,
                         name='output',
                         activation='relu',
                         kernel_initializer='he_normal')(concat)

    model = models.Model(inputs=[inputA, inputB, inputC], outputs=[final])
    model.compile('rmsprop', 'mse', metrics=['accuracy'])
    hist = model.fit([ing, about, category], score, epochs=50, batch_size=16)

    res = model.evaluate([ing, about, category], score)
    pred = model.predict([ing, about, category])
    print(pred)
    plt.plot(pred, label='pred')
    plt.plot(score, label='score')
    plt.grid()
    plt.legend()
    plt.show()

    return model


def ing_model(ing, score) -> None:
    model = models.Sequential()
    model.add(
        layers.Dense(56,
                     activation='relu',
                     input_shape=(56, ),
                     kernel_initializer='he_normal'))
    model.add(
        layers.Dense(200, activation='relu', kernel_initializer='he_normal'))
    model.add(
        layers.Dense(100, activation='relu', kernel_initializer='he_normal'))
    model.add(
        layers.Dense(1, activation='relu', kernel_initializer='he_normal'))
    model.compile('rmsprop', 'mse', metrics=['accuracy'])
    hist = model.fit([ing], score, epochs=50, batch_size=1024)

    return model


def new() -> None:

    with open('dataset.csv', newline='') as csvfile:
        spamreader = csv.DictReader(csvfile)
        name = [row['name'] for row in spamreader]

    with open('dataset.csv', newline='') as csvfile:
        spamreader = csv.DictReader(csvfile)
        ing = [row['ing'] for row in spamreader]

    with open('dataset.csv', newline='') as csvfile:
        spamreader = csv.DictReader(csvfile)
        about = [row['about'] for row in spamreader]

    with open('dataset.csv', newline='') as csvfile:
        spamreader = csv.DictReader(csvfile)
        category = [row['category'] for row in spamreader]

    with open('dataset.csv', newline='') as csvfile:
        spamreader = csv.DictReader(csvfile)
        score = [row['score'] for row in spamreader]

    arr = np.array([name, category, about, ing, score])

    about = np.array(
        [np.array(text.hashing_trick(arr[2, i], 91)) for i in range(357)])
    ing = np.array(
        [np.array(text.hashing_trick(arr[3, i], 56)) for i in range(357)])

    ing = sequence.pad_sequences(ing, maxlen=56, padding='post')
    about = sequence.pad_sequences(about, maxlen=91, padding='post')

    hist = {}
    category = set(category)

    for i in enumerate(category):
        hist[i[1]] = i[0]

    category = np.array(
        [np.array(to_categorical(hist[arr[1, i]], 19)) for i in range(357)])

    score = np.array([float(x) for x in score])

    pickle.dump((about, ing, category, hist), open('points.pkl', 'wb'))

    mdl = new_model(about, ing, category, np.array(score))
    mdl = models.load_model('model.h5')
    # mdl.save('model.h5')


def pipeline() -> None:
    about = input()
    ing = input()
    category = input()
    points = pickle.load(open('points.pkl', 'rb'), encoding='utf-8')
    mdl = models.load_model('model.h5')

    about = np.array(text.hashing_trick(preprocess(about), 91))
    ing = np.array(text.hashing_trick(preprocess(ing), 56))
    category = np.array(to_categorical(points[-1][category], 19))
    ing = sequence.pad_sequences(ing, maxlen=56, padding='post')
    about = sequence.pad_sequences(about, maxlen=91, padding='post')
    mdl.predict([ing, about, category])


if __name__ == '__main__':
    pipeline()
