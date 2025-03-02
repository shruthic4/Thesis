import csv
from sklearn import metrics
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def remove_outliers_iqr(y):
    mask = np.ones(len(y), dtype=bool)  # Start with all rows as valid
    
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask &= (y >= lower_bound) & (y <= upper_bound)
    return mask

def calc_correlations(X,y):
    correlations = X.corrwith(y) 
    print(*correlations, sep = '\n')
  

def bin_data(y):
    bins = np.linspace(y.min(), y.max(), 6)
    y_binned = np.digitize(y, bins, right=False) - 1
    y_binned[y_binned == 5] = 4
    return y_binned

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # use R² score, the coefficient of determination --> value of 0.85 is a good cutoff value
    r2_score = metrics.r2_score(y_test, y_pred)
    print("R^2 score of model", r2_score)
    return r2_score

    
def parse():
    parser = argparse.ArgumentParser(
                    prog='ML model ',
                    description='Trains ML model for predicting NIH cognition scores based on neural data',
                    )
    parser.add_argument('subjects', help='HOA - healthy older adults or OA- healthy older adults and MCI ')
    parser.add_argument('cog_type', help='s - speed, c - crystallized intelligence, m - memory')
    return parser.parse_args().subjects, parser.parse_args().cog_type