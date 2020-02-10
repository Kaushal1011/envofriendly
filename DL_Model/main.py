#!/usr/bin/env python3

import pickle
from pprint import pprint

import pandas as pd
from tqdm import trange


def fetch_india() -> None:
    # df = pd.read_csv('en.openfoodfacts.org.products.tsv',
    #                  sep='\t',
    #                  encoding='utf-8')
    with open('df.pkl', 'rb') as f:
        df: pd.DataFrame = pickle.load(f, encoding='utf-8')

    india = []
    for i in trange(len(df)):
        if df.iloc[i]['countries_en'] == 'India':
            india.append(df.iloc[i])

    with open('india.pkl', 'wb') as f:
        pickle.dump(india, f)

    print(len(india))


def main() -> None:
    df = pd.read_csv('GHG - Sheet1.csv', encoding='utf-8')
    # df.drop('Unnamed: 15', inplace=True, axis=1)
    print(df)


if __name__ == '__main__':
    main()
