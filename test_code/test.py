# system
import os

# basic
import numpy as np
import pandas as pd

# visual
import matplotlib.pyplot as plt

# sklearn evaluate
from sklearn import metrics


train_data = pd.read_csv('training.csv',sep = ',', names=["Open","High","Low","Close"])
test_data = pd.read_csv('testing.csv',sep = ',', names=["Open","High","Low","Close"])
print(train_data)
print(test_data)
