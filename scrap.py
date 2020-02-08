#!/usr/bin/env python3

import json
from pprint import pprint

import bs4
import requests
from tqdm import tqdm


def scrap() -> None:
    urls = [298369, 10000148, 10000153]

    baseurl = 'https://www.bigbasket.com/pd/'
    items = {}
    for i in tqdm(urls):
        url = baseurl + str(i)
        page = requests.get(url)

        soup = bs4.BeautifulSoup(page.content, 'html.parser')

        res = {}
        res['name'] = soup.find('h1', class_='GrE04').text.strip()
        res['price'] = soup.find('td',
                                 class_='IyLvo').text.split('Rs')[-1].strip()
        res['img'] = soup.findAll('img', class_='_3oKVV')[-1]['src']
        res['about'] = soup.find(id='about_0')
        res['ing'] = soup.find(id='about_1')

        try:
            res['about'] = soup.find('div', class_='_26MFu').find('div').text

            if res['ing'].find('span').text != 'INGREDIENTS':
                res['ing'] = res['name']
            else:
                res['ing'] = res['ing'].find('div',
                                             class_='_26MFu').find('div').text
        except AttributeError:
            continue

        items[res['name']] = res

    pprint(items)


if __name__ == '__main__':
    scrap()
