import csv
from os import read
import numpy as np 
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# generate feature vector

# normalize features

# create classifier with specified parameters (similar to select param_logreg -based off of cv performance)


def data_preproc():
    Hoa_data = pd.read_csv("shruthi_HOA_mastersheet.csv")
    # store features
    X = Hoa_data[Hoa_data.columns[2:5]]
    # store labels 
    y = Hoa_data[Hoa_data.columns[5:]]
    
    # TODO: add stratification - y_binned
    return train_test_split(X, y, test_size=0.20, random_state= 4)



def train_lasso(X_train, y_train):
   
    # Change alpha to tweak regularization strength (increase alpha --> shrinks feature coeffs to zero  --> decrease complexity)
    # TODO: use CV to select best alpha 
    model = Lasso
    model.fit(X_train, y_train)
    return model 

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # performance metric, how to evaluate? - how close is y_pred to y_score
    # use mean absolute error 
    # y is standardized, so MAE would indicate how many standard deviations away y_pred is to y_true
    # Calculate MAE for each class, 
 




def main():
    X_train, X_test, y_train, y_test = data_preproc()
    model = train_lasso(X_train, y_train)
    breakpoint()
    evaluate(model, X_test, y_test)
    
   
    print("test")


if __name__ == "__main__":
    main()