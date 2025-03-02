
import csv
from os import read
import numpy as np 
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from scipy.linalg import svd
from sklearn.inspection import permutation_importance
from regression_HOA import generate_coeff_plot
from utils import bin_data, evaluate, remove_outliers_iqr, parse, calc_correlations

def data_preproc():
    subj_data = pd.read_csv(f"shruthi_new_OA_mastersheet.csv")
    # Add columns for labels: 0= HOA, MCI = 1
    subj_data['Label'] =  subj_data['Subject'].str.startswith('mindm').astype(int)
    # randomly sample 30 HOA to use for analysis 

    # ALL DATA
    # X = subj_data[subj_data.columns[5:-1]]
    # classify young v old
    y =  subj_data['Label']
    

    
    # try with just volume data
    #X = subj_data[subj_data.columns[149:272]]
    # try just metabolite
   # X = subj_data[subj_data.columns[5:149]]
    # store features - don't include distinctivenns
    #41 (conn data)
    X =  subj_data[ subj_data.columns[5:273] ]

    #X = subj_data.iloc[:, np.r_[5:41, 273:318]]
    X = X.applymap(lambda val: float(val.split('+')[0]) if isinstance(val, str) and '+0i' in val else val)

    
    # stratify data
    # y_binned = bin_data(y)
    # unique, counts = np.unique(y_binned, return_counts=True)
    # print("Bin distribution:", dict(zip(unique, counts)))
    # print(y_binned)
    # return train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)
    return  X,y 



def train_log_reg(X_train, y_train):
   
    # Change alpha to tweak regularization strength (increase alpha --> shrinks feature coeffs to zero  --> decrease complexity)
    # C = inverse of alpha
    # use CV to select best alpha 

    cv = StratifiedKFold(n_splits=3, shuffle=True)
    # Train (9-10), Val (4-5)
    param_grid = {'C': np.logspace(-4, 2, 100)} 
    # Set up GridSearchCV
    grid = GridSearchCV(
        estimator=LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced'),
        param_grid=param_grid,
        scoring='roc_auc',  
        cv=cv,  #  cross-validation
        return_train_score=True
)
    grid.fit(X_train, y_train)

    # Extract training scores
    train_scores = grid.cv_results_['mean_train_score']
    validation_scores = grid.cv_results_['mean_test_score']
    Cs = grid.cv_results_['param_C']
# Plot training and validation scores vs. regularization strength
    plt.figure(figsize=(10, 6))
    plt.plot(Cs, train_scores, label="Training Score", marker='o', linestyle='-', color='blue')
    plt.plot(Cs, validation_scores, label="Validation Score", marker='o', linestyle='--', color='orange')
    plt.xscale('log')  
    plt.xlabel('C (Inverse Regularization Strength)')
    plt.ylabel('AUROC Score')
    plt.title('Training and Validation Scores vs. C')
    plt.legend()
    plt.grid(True)
    # plt.show()

    print("Optimal C:", grid.best_params_['C'])

    # Train final model with best C
    model = LogisticRegression(penalty='l1', solver='liblinear', C=grid.best_params_['C'], class_weight= 'balanced')
    model.fit(X_train, y_train)

    return model

# def train_log_reg(X_train, y_train):
    

def train_svm(X_train,y_train):
    
    # gamma: low(smooth DB, points influence larger area), high(sensitive DB, points influence smaller area)
    # C- scalar on the the loss function--> A larger C (more penalities on misclassifications), smaller c--> more penalty on margin

    param_grid = {
    'C': [1,  10, 100, 1000, 10000, 100000],
    'kernel': ['linear'],
    'gamma': ['scale', 'auto']
}
    # 15 MCI and 15 HOA - 12-13(train)  & 2-3(val)
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    grid = GridSearchCV(estimator=SVC(class_weight= 'balanced'), param_grid=param_grid, 
                           cv=cv,scoring='roc_auc', verbose=2, return_train_score=True)

    # Perform the grid search on training data
    grid.fit(X_train, y_train)

   

    ### Create a 2D plot
    # # Extract values for 2D plotting
    # results_df = pd.DataFrame(grid.cv_results_)
    # C_vals = results_df['param_C'].astype(float)
    # val_scores = results_df['mean_test_score']
    # train_scores = results_df['mean_train_score']

    # plt.figure(figsize=(10, 7))

    # # Plot training scores
    # plt.plot(C_vals, train_scores, label="Training Score", marker='o', linestyle='-', color='blue')

    # # Plot validation scores
    # plt.plot(C_vals, val_scores, label="Validation Score", marker='o', linestyle='-', color='orange')

    # # Axis labels and title
    # plt.xlabel("C Values")
    # plt.ylabel("Score")
    # plt.title("Training vs Validation Scores (2D Grid Search)")

    # # Add a legend
    # plt.legend()

    # # Show plot
    # plt.show()

    # Print the best parameters and score
    print("Best parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)


    # train SVR based on params from best CV results
    best_model = SVC(**grid.best_params_, class_weight='balanced')

    best_model.fit(X_train, y_train) 

   
    return best_model



def train_rf(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    # Train (9-10), Test (4-5)
    cv = StratifiedKFold(n_splits=3, shuffle=True,)
    
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1  # Optional: Shows progress
    )

    grid_search.fit(X_train, y_train)

    # Best parameters and score
    print("\nBest parameters:", grid_search.best_params_)
    print("Best AUROC score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    return best_model






def custom_importance_getter(estimator, X, y):
    result = permutation_importance(estimator, X, y, n_repeats=5, random_state=42)
    return result.importances_mean  # Mean importance of each feature

def custom_rfe_with_plot(X, y, model, n_splits=3):
    n_features = X.shape[1]
    feature_indices = np.arange(n_features)  # Keep track of feature indices
    feature_rankings = np.zeros(n_features)  # To store rankings for features
    performance_scores = []  # To store average performance at each iteration
    remaining_features = []  # Track number of features at each step
    # breakpoint()
    while len(feature_indices) > 1:
        # print(f"Running with {len(feature_indices)} features...")
        fold_weights = np.zeros((n_splits, len(feature_indices)))
        fold_scores = []  # Track scores for current feature subset
        stratified_cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
        for i, (train_idx, val_idx) in enumerate(stratified_cv.split(X, y)):
            # Train and evaluate model on current subset of features
            X_train, X_val = X.iloc[train_idx, feature_indices], X.iloc[val_idx, feature_indices]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train, y_train)
            y_pred = model.decision_function(X_val)
            score = metrics.roc_auc_score(y_val, y_pred)
            fold_scores.append(score)  # Calculate and save auroc 

            # Capture feature weights
            if hasattr(model, "coef_"):
                fold_weights[i] = np.abs(model.coef_).flatten()
            else:
                fold_weights[i] = np.abs(model.dual_coef_).mean(axis=0)

        # Average the feature weights and performance scores across folds
        avg_weights = fold_weights.mean(axis=0)
        avg_auroc_score = np.mean(fold_scores)

        # Track performance for this subset of features
        performance_scores.append(avg_auroc_score)
        remaining_features.append(len(feature_indices))

        # Identify and remove the least important feature
        least_important_index = np.argmin(avg_weights)
        # print(f"Removing feature {feature_indices[least_important_index]} with weight {avg_weights[least_important_index]}")
        # assign ranking based on what number it was removed - higher ranking --> removed earlier
        feature_rankings[feature_indices[least_important_index]] = len(feature_indices)
        feature_indices = np.delete(feature_indices, least_important_index)
       

    # Final feature gets rank 1
    feature_rankings[feature_indices[0]] = 1
    performance_scores.append(avg_auroc_score)
    remaining_features.append(len(feature_indices))
    best_perf = max(performance_scores)
    n_features_to_select = remaining_features[performance_scores.index(best_perf)]
    print(f"Peak performance of {best_perf} at: {n_features_to_select}" )

    # hard set n_features to 75
    # n_features_to_select = 75
    ### Plot performance vs. number of features
    # plt.figure(figsize=(10, 6))
    # plt.plot(remaining_features, performance_scores, marker='o')
    # plt.gca().invert_xaxis()  # Invert x-axis to show fewer features on the right
    # plt.title("Model Performance vs. Number of Features")
    # plt.xlabel("Number of Features")
    # plt.ylabel("Average Auroc Score")
    # plt.grid()
    # plt.show()
    

    # sort features and take the top N feaures to return as the new DF
    index_to_rank = np.argsort(feature_rankings)
    X_ranked = X.columns[index_to_rank]

    return X_ranked, n_features_to_select
def RFE_SVM(X,y):
    # X and Y are the entire set 
    resampler = RandomUnderSampler()
    X,y = resampler.fit_resample(X, y)
    # outer CV for model performance evaluation
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True)
    outer_scores = []  # Store AUROC scores from each outer fold
    # NOTE: features that don't show up will be defaulted to 0 for each fold
    # Track selected feature WEIGHTS across all folds
    feature_importances_df = pd.DataFrame(0, index=range(3), columns=X.columns, dtype=float)
    selected_features_list = []  # Track selected features across all folds

    # Train: 15  - Test: 7
    with open("classification_reports.csv", "w") as f:
        f.write("Fold,Metric,Precision,Recall,F1-Score,Support\n")  # CSV Header

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Initialize the scaler
        scaler = StandardScaler()

        # Fit only on training data
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)

        # Transform the test data using the same scaler
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

        model = SVC(kernel = 'linear', class_weight= 'balanced' )
        ranked_X, top_n= custom_rfe_with_plot(X_train, y_train,model)
        X_train_selected = X_train[ranked_X[:top_n]]
        X_test_selected = X_test[ranked_X[:top_n]]

    
        ### MODEL = RF/log_reg =--> predict proba, SVC = decision func

        # NOTE: Model trained on features selected from RFE on training set 
        # Trained on 15 MCI/15 HOA

        ## svm with RFE
        model = train_svm(X_train_selected, y_train)
        y_prob = model.decision_function(X_test_selected)
    

        ## log_reg with RFE
        # model = train_log_reg(X_train_selected, y_train)
        # y_prob = model.predict_proba(X_test_selected)[:, 1]
    
        ## RF with RFE  
        # model = train_rf(X_train_selected,y_train)
        # y_prob = model.predict_proba(X_test)[:, 1]

        auroc = metrics.roc_auc_score(y_test, y_prob)
        outer_scores.append(auroc)

        ## SVM/ log_reg
        # NOTE: After training model on selected features from RFE - get coeffs 
        selected_coeffs = pd.Series(model.coef_.flatten(), index=ranked_X[:top_n])
        

        ## RF 
        #selected_coeffs = pd.Series(model.feature_importances_, index=ranked_X[:top_n])

        feature_importances_df.loc[fold, selected_coeffs.index] = selected_coeffs

        selected_features_list.append(ranked_X[:top_n])
        print("Auroc score of model", auroc)
        y_pred = model.predict(X_test_selected)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print(metrics.classification_report(y_test, y_pred))

  # NOTE: save results to csv - easy to export
        # Convert classification report to a dictionary
        with open("classification_reports.csv", "a") as f:  # "a" ensures results are appended
            report_dict = metrics.classification_report(y_test, y_pred, output_dict=True)
            for metric, values in report_dict.items():
                    if isinstance(values, dict):  # Ignore "accuracy" since it's a single value
                        f.write(f"{"Run"},{metric},{values['precision']},{values['recall']},{values['f1-score']},{values['support']}\n")

            # Manually add accuracy to CSV
            f.write(f"{"Run"},accuracy,,,{accuracy},\n")

            f.write(f"{"Run"},AUROC,,,{auroc},\n")
        



    # Feature Selection Frequency
    feature_counts = pd.Series([item for sublist in selected_features_list for item in sublist]).value_counts()

    table = PrettyTable()
    table.field_names = ["Feature", "Count"]

    # Populate the table
    for feature, count in feature_counts.items():
        table.add_row([feature, count])  

    # # Display the table
    # print(table)

    # Generate plot for avg coeffs during each fit of the model
    
    # X vals 
    gen_avg_coeff_table(feature_importances_df.values, X)


    print("\nAverage Test AUROC across Outer Folds:", np.mean(outer_scores))
    print("Standard Deviation of Test AUROC:", np.std(outer_scores))

def CV_model_eval(X,y):
    with open("classification_reports.csv", "w") as f:
        f.write("Fold,Metric,Precision,Recall,F1-Score,Support\n")  # CSV Header
        resampler = RandomUnderSampler()
        X,y = resampler.fit_resample(X, y)
        outer_cv = StratifiedKFold(n_splits=3, shuffle=True)
        # Train (15), Test(7)
        outer_scores = []  # Store AUROC scores from each outer fold
        feature_importances = []  # Track selected features across all folds
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            # Initialize the scaler
            scaler = StandardScaler()

            # Fit only on training data
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)

            # Transform the test data using the same scaler
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

            # TODO check distribution 
            ## SVM
            # model = train_svm(X_train, y_train)
            # y_prob = model.decision_function(X_test)
        
            ## log_reg with RFE
            # model = train_log_reg(X_train, y_train)
            # y_prob = model.predict_proba(X_test)[:, 1]
        
            ## CV w RF - no RFE: 
            model = train_rf(X_train,y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            auroc = metrics.roc_auc_score(y_test, y_prob)
            y_pred = model.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)    
            print(metrics.classification_report(y_test, y_pred))
            # Keep track of each run's auroc 
            outer_scores.append(auroc)
            # TODO: keep track of classificication report - averages --> manually calc averages

            # NOTE: save results to csv - easy to export
            # Convert classification report to a dictionary
            report_dict = metrics.classification_report(y_test, y_pred, output_dict=True)
            for metric, values in report_dict.items():
                    if isinstance(values, dict):  # Ignore "accuracy" since it's a single value
                        f.write(f"{"Run"},{metric},{values['precision']},{values['recall']},{values['f1-score']},{values['support']}\n")

            # Manually add accuracy to CSV
            f.write(f"{"Run"},accuracy,,,{accuracy},\n")

            f.write(f"{"Run"},AUROC,,,{auroc},\n")


            ## Random Forest --> Feature importance is based on Gini Impurity
            # "The higher, the more important. The importance of a feature is computed as the
            # (normalized) total reduction of the criterion brought by that feature." 
            feature_importances.append(model.feature_importances_)


            ## SVM/Log_reg --> feature importance is based on magnitude of feature coefs
             # feature_importances.append(model.coef_.flatten())

        
            
            print("Auroc score of model", auroc)

   
    # Generate plot for avg coeffs during each fit of the model
    gen_avg_coeff_table(feature_importances, X)
    print("\nAverage Test AUROC across Outer Folds:", np.mean(outer_scores))
    print("Standard Deviation of Test AUROC:", np.std(outer_scores))
    

    # TODO: check if this function is matching cols w data correctly 
    
def gen_avg_coeff_table(feature_importances, X, filename="results.csv"):
      # Convert to DataFrame
    coef_df = pd.DataFrame(feature_importances, columns=X.columns)
    mean_coefs = coef_df.mean()
    std_coefs = coef_df.std()

    # Combine into a DataFrame for sorting
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Avg Coefficient': mean_coefs.values,
        'Std Dev': std_coefs.values
    })

    # Add column for absolute value of average coefficient for sorting
    importance_df['Abs Avg Coefficient'] = importance_df['Avg Coefficient'].abs()

    # Sort by absolute value of average coefficient in descending order
    importance_df.sort_values(by='Abs Avg Coefficient', ascending=False, inplace=True)


    # Save feature importances to CSV
    importance_df.drop(columns=['Abs Avg Coefficient']).to_csv(filename, index=False)

    # Create PrettyTable
    table = PrettyTable()
    table.field_names = ["Feature", "Avg Coefficient", "Std Dev"]

    # Populate table with sorted values
    for _, row in importance_df.iterrows():
        table.add_row([row['Feature'], round(row['Avg Coefficient'], 4), round(row['Std Dev'], 4)])

    # Display the table
    print("\nFeature Importance (Sorted by Abs Avg Coefficient):\n")
    print(table)

    return table
   

  

def main():
    X,y = data_preproc()
    #RFE_SVM(X,y)
    CV_model_eval(X,y)
 

    # plt.hist(y_train, bins=10, alpha=0.5, label='Train')
    # plt.hist(y_test, bins=10, alpha=0.5, label='Test')
    # plt.legend()
    # plt.show()

    # features = train_lasso(X_train, y_train)
    # print(features)
    # try auto encoders, SVD, clustering

    # only use selected features to train svr
    # X_train = X_train
    # model = train_svr(X_train, y_train)
    # ranked_X, top_n= custom_rfe_with_plot(X_train, y_train,model)
    # breakpoint()
    # X_train_selected = X_train[ranked_X[:top_n]]
    # # # model = train_svr(X_train_selected, y_train)
    # X_test_selected = X_test[ranked_X[:top_n]]
    # model = train_ridge(X_train_selected, y_train)

    ####
    # _,  model = train_log_reg(X_train, y_train)
    
    # breakpoint()

    # y_pred = model.predict(X_test)
    # auroc = metrics.roc_auc_score(y_test, y_pred)
    # print(metrics.classification_report(y_test, y_pred))
    # print(auroc)
    # # print(f"R^2 score on Test Data (after tuning): {test_r2_best}")
    # # print("R^2 score of model", r2_score)
    # plt.scatter(y_test, y_pred, alpha=0.5)
    # plt.xlabel("Actual")
    # plt.ylabel("Predicted")
    # plt.title("Predicted vs. Actual")
    # plt.show()

    # ####
    

    # # model = SVC(kernel = 'linear')
    # model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
    # ranked_X, top_n= custom_rfe_with_plot(X_train, y_train,model)
    # X_train_selected = X_train[ranked_X[:top_n]]
    # # model = train_svr(X_train_selected, y_train)
    # X_test_selected = X_test[ranked_X[:top_n]]
    # model = train_svm(X_train_selected, y_train)


    # y_pred = model.predict(X_test_selected)
    # auroc = metrics.roc_auc_score(y_test, y_pred)
    # # print(f"R^2 score on Test Data (after tuning): {test_r2_best}")
    # print("auroc score of model", auroc)
    

    # plt.scatter(y_test, y_pred, alpha=0.5)
    # plt.xlabel("Actual")
    # plt.ylabel("Predicted")
    # plt.title("Predicted vs. Actual")
    # plt.show()
    # # print(y_test)
    # # print(y_pred)
    # breakpoint()
    # print(metrics.classification_report(y_test, y_pred))

    
    
    
   


if __name__ == "__main__":
    main()
