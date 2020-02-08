#!/usr/bin/env python3

import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def main() -> None:
    ghg = pd.read_csv('GHG - Sheet1.csv')
    ghg = ghg[['croplivestock-prod.', 'GHG-[kg-kg−1-product]']]

    # pprint(ghg)

    with open('items.pkl', 'rb') as f:
        df = pd.DataFrame(pickle.load(f, encoding='utf-8'))
        df = df.T

    # pprint(df.loc['Fresho Onion, 1 kg', 'ing'])
    # pprint(df.index)
    # for i in df.index:
    #     print(type(df.loc[i, 'ing']))

    # pprint(ghg)

    ings = []
    for i in df.index:
        ing = df.loc[i, 'ing']
        score = 0
        cnt = 0
        for j in ghg['croplivestock-prod.']:
            if j.lower() in ing.lower():
                cnt += 1
                score += j['GHG-[kg-kg−1-product]']

        ings.append(score / cnt)

    pprint(ings)


def clean() -> None:
    df = pd.DataFrame(pickle.load(open('items.pkl', 'rb'), encoding='utf-8'))
    df = df.T

    eco_num = pd.read_csv('GHG - Sheet2.csv')

    cnts = []
    for i in df.index:
        cnt = 0
        for j in eco_num['words']:
            if j.lower() in df.loc[i, 'about'].lower():
                cnt += 1
        cnts.append(cnt)

    print(cnts)
    print(df['category'])
    plt.plot(range(len(cnts)), cnts)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    clean()
