#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:12:22 2018

@author: kuprina_laptop
"""
import os as os
import pandas as pd
import numpy as np
from fancyimpute import KNN
import matplotlib.pyplot as plt

os.chdir('/Users/kuprina_laptop/Documents/titanic/')

data_test = pd.read_csv('test.csv')
data_train = pd.read_csv('train.csv')

df = data_train #data_test.append(data_train)
df.describe() ## instead of summary()

## quotation marks are necessary

# how big is the dataset?
df.shape[0]

# how many are NAs?
df.isnull().sum()

# % of NaN from total
df.isnull().sum()/df.shape[0] * 100

#for names in df["Name"]:
#    print(names.split(',')[1].split("."))

#df['Name'].str.split(',').str.split(".")[0]


df['Title'] = df.Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False)

df_t = df.select_dtypes(include = [np.number])
df_filled = pd.DataFrame(KNN(k=5).fit_transform(df.select_dtypes(include = [np.number]).as_matrix()))

df_filled.columns = df_t.columns

df['Age'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.grid(axis='y', alpha=0.75)


## interpolate Cabin variable using sex, age, fare, title
## encode title and other string cols
## 


