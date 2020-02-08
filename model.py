#!/usr/bin/env python3

import re
import string
from pprint import pprint
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import text
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


def create_word_index(df: pd.DataFrame, vocab: str) -> Dict[str, float]:
    '''Return 0 to 1 index of vocabulary'''

    words = [x.split() for x in df[vocab]]

    res = []
    [res.extend(x) for x in words]
    res = set(res)

    index = {}
    for i in enumerate(res, start=1):
        # index[i[1]] = i[0] / len(res)
        index[i[1]] = i[0]

    return index


def model(train: pd.DataFrame) -> models.Sequential:
    about_model = models.Sequential()
    about_model.add(
        layers.Embedding(input_dim=91, output_dim=200, input_length=91))
    about_model.add(layers.Dense(100, activation='elu'))

    ing_model = models.Sequential()
    ing_model.add(
        layers.Embedding(input_dim=56, output_dim=120, input_length=56))
    ing_model.add(layers.Dense(100, activation='elu'))

    merge_init = models.Sequential()
    merge_init.add(layers.Concatenate([about_model, ing_model]))
    merge_init.add(layers.Dense(100, activation='elu'))

    cat_model = models.Sequential()
    cat_model.add(layers.Dense(19, activation='elu'))

    merge_init1 = models.Sequential()
    merge_init1.add(layers.Concatenate([merge_init, cat_model]))

    merge_init1.add(layers.Dense(1, activation='elu'))

    # about_model = layers.Input(shape=(91, ), name='about_model')
    # ing_model = layers.Input(shape=(56, ), name='ing_model')

    model = models.Model(inputs=[about_model, ing_model, cat_model],
                         outputs=[merge_init1])

    model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])

    plot_model(model, to_file='model.png')

    return model


def main() -> None:
    df = pd.read_csv('dataset.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)

    about_index = create_word_index(df, 'about')

    num = max([len(x.split()) for x in df['about']])
    print(num)
    L = []
    for i in range(df.shape[0]):
        l: List[str] = df.iloc[i]['about'].split()
        for j in enumerate(l):
            l[j[0]] = about_index[j[1]]
        l.extend([0] * (num - len(l)))
        L.append(l)

    df = pd.DataFrame({
        'name': df['name'],
        'category': df['category'],
        'score': df['score'],
        'ing': df['ing'],
        'about': L
    })

    ing_index = create_word_index(df, 'ing')
    num = max([len(x.split()) for x in df['ing']])
    print(num)
    L = []
    for i in range(df.shape[0]):
        l: List[str] = df.iloc[i]['ing'].split()
        for j in enumerate(l):
            l[j[0]] = ing_index[j[1]]
        l.extend([0] * (num - len(l)))
        L.append(l)

    df = pd.DataFrame({
        'name': df['name'],
        'category': df['category'],
        'score': df['score'],
        'ing': L,
        'about': df['about']
    })

    hist = {}
    for i in enumerate(set(df['category'])):
        hist[i[1]] = i[0]
    L = []
    for i in range(df.shape[0]):
        l = to_categorical(hist[df.iloc[i]['category']], num_classes=len(hist))
        L.append(l)

    df = pd.DataFrame({
        'name': df['name'],
        'category': L,
        'score': df['score'],
        'ing': df['ing'],
        'about': df['about']
    })

    rand_df = pd.DataFrame(np.random.randn(100, 2))
    msk = np.random.rand(len(df)) < 0.8

    train = df[msk]
    test = df[~msk]

    # pprint(train)
    # pprint(test)

    model(train)


if __name__ == '__main__':
    main()
