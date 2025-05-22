import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# LOad the dataset
df = pd.read_csv(r"C:\Users\uchen\Downloads\cmu-sleep.csv")

# Data Overview
print(df.head(5))

# Data information
print(df.info())

# Statistical summary
print(df.describe())

# Check for missing value
print(df.isnull().sum())

