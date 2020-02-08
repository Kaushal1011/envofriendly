#!/usr/bin/env python3

import re
import string
from pprint import pprint

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


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


def create_word_index(df: pd.DataFrame) -> dict:
    words = [x.split() for x in df['about']]

    res = []
    [res.extend(x) for x in words]
    res = set(res)

    index = {}
    for i in enumerate(res):
        index[i[1]] = i[0]

    return index


def main() -> None:
    df = pd.read_csv('dataset.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    # df = scale_scores(df)

    word_index = create_word_index(df)

    pprint(word_index)
    pprint(df.columns)


if __name__ == '__main__':
    main()
