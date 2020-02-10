import csv
import json
from pprint import pprint
import pandas as pd


# csvfile = open('envofriendly-scraping/dataset.csv', 'r')

def convert(name, price, imgurl, about, ing, category):
    string = "mutation { addProduct (name: \""
    string += str(name)
    string += "\", about: \""
    string += str(about)
    string += "\", price: "
    string += str(price)
    string += ", category: \""
    string += str(category)
    string += "\", ing: \""
    string += str(ing)
    string += "\", imageurl: \""
    string += str(imgurl)
    string += "\") { id } }"
    print(string)

df = pd.read_csv('items.csv')

for i in range(len(df)):
    name = df.iloc[i]['name']
    price = df.iloc[i]['price']
    img = df.iloc[i]['img']
    about = df.iloc[i]['about']
    about = about.replace("\n"," ")
    about = about.replace("\t"," ")
    about = about.lstrip();
    about = about.rstrip();
    ing = df.iloc[i]['ing']
    category = df.iloc[i]['category']
    convert(name, price, img, about, ing, category)
