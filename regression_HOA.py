
import csv
from os import read
import numpy as np 
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from scipy.linalg import svd
from sklearn.inspection import permutation_importance
from utils import bin_data, evaluate, remove_outliers_iqr, parse, calc_correlations
# TODO: create class - reuse vars
def data_preproc(group, y):
    subj_data = pd.read_csv(f"shruthi_new_OA_mastersheet.csv")
    X = subj_data[subj_data.columns[5:]]
    # Till 41 is just conn
    breakpoint()
    # try with just volume data
    #X = subj_data[subj_data.columns[108:237]]
    # try just metabolite
    #X = subj_data[subj_data.columns[5:108]]
    # store features - don't include distinctivenns
    #X =  subj_data[ subj_data.columns[5:237]]
    X = X.applymap(lambda val: float(val.split('+')[0]) if isinstance(val, str) and '+0i' in val else val)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # X = try_svd(X)
    # breakpoint()
    # determine which model to train
    if y == 's': # speed
        y =   subj_data[subj_data.columns[2]]
    elif y == 'c': # crystallized 
        y  =  subj_data[subj_data.columns[3]]
    else: # memory
        y =  subj_data[subj_data.columns[4]]   
    # breakpoint()


  
    mask = remove_outliers_iqr(y)
    X, y = X[mask], y[mask]

    # generate correlation matrix
    calc_correlations(X,y)

    # plt.hist(y, bins=20)
    # plt.title("Y value distribution")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.show()

    # stratify data
    y_binned = bin_data(y)
    # unique, counts = np.unique(y_binned, return_counts=True)
    # print("Bin distribution:", dict(zip(unique, counts)))
    # print(y_binned)
    # return train_test_split(X, y, test_size=0.2, random_state=4, stratify=y_binned)
    return train_test_split(X, y, test_size=0.2, random_state=42)


    

def try_svd(X):
    svd = TruncatedSVD(n_components=min(X.shape) - 1, random_state=42)
    svd.fit(X)

    # Cumulative explained variance
    cumulative_variance = np.cumsum(svd.explained_variance_ratio_)

    # Find the number of components for desired variance (e.g., 95%)
    target_variance = 0.95
    n_components = np.argmax(cumulative_variance >= target_variance) + 1
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    return X_reduced



def generate_coeff_plot(coefficients, features):
     # Create a bar plot
    coeffs_plot = coefficients[coefficients != 0]
    features = features[coefficients != 0]
    table = PrettyTable()
    # breakpoint()
    # pretty print the table
    table.field_names = ["Feature", "Coefficient"]
    for feature, coef in zip(features, coeffs_plot):
        table.add_row([feature,coef])  
   
    print(*features, sep= '\n')
    print(*coeffs_plot, sep='\n')
    print(table)   

    plt.figure(figsize=(40, 6))
    plt.bar(features, coeffs_plot, color='skyblue')
    plt.axhline(0, color='black', linewidth=0.1,  linestyle='--')
    plt.title('Lasso Coefficients')
    plt.xlabel('Features')
    #breakpoint()
    plt.xticks(rotation=45, fontsize=5) 
    plt.ylabel('Coefficient Value')
    plt.show()



def train_lasso(X_train, y_train):
   
    # Change alpha to tweak regularization strength (increase alpha --> shrinks feature coeffs to zero  --> decrease complexity)
    # use CV to select best alpha 

    param_grid = {'alpha': np.logspace(-4, 2, 100)} 
    breakpoint()
    # Set up GridSearchCV
    grid = GridSearchCV(
        estimator=Lasso(),
        param_grid=param_grid,
        scoring='r2',  
        cv=5,  # 5-fold cross-validation
        return_train_score=True
)
    grid.fit(X_train, y_train)

    # Extract training scores
    train_scores = grid.cv_results_['mean_train_score']
    validation_scores = grid.cv_results_['mean_test_score']
    alphas = grid.cv_results_['param_alpha']

    # plot train, test, and parameters
    plt.figure(figsize=(10, 6))

    # Plot training scores
    plt.plot(alphas, train_scores, label="Training Score", marker='o', linestyle='-', color='blue')

    # Plot validation scores
    plt.plot(alphas, validation_scores, label="Validation Score", marker='o', linestyle='--', color='orange')

    # Customize the plot
    plt.xscale('log')  # Alphas are usually plotted on a log scale
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('r2 score')
    plt.title('Training and Validation Scores vs. Alpha')
    plt.legend()
    plt.grid(True)
    plt.show()

    # print("Optimal alpha:", lasso_cv.alpha_)
    # #Optimal alpha: 0.21314946588529884

    model = Lasso(alpha = grid.best_params_['alpha'])
    model.fit(X_train, y_train)

    coefficients = model.coef_
    print(coefficients)
    #breakpoint()
    features = X_train.columns
    generate_coeff_plot(coefficients, features)
    features = features[coefficients != 0]

    return features

def train_svr(X_train,y_train):
    
    # gamma: low(smooth DB, points influence larger area), high(sensitive DB, points influence smaller area)
    # C- scalar on the the loss function--> A larger C (more penalities on misclassifications), smaller c--> more penalty on margin

    param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [ 0.1, 0.2, 0.5],
    'kernel': [ 'rbf', 'linear'],
    'gamma': ['scale', 'auto']
}
    grid = GridSearchCV(estimator=SVR(), param_grid=param_grid, 
                           cv=5, scoring='r2', verbose=2, return_train_score=True)

    # Perform the grid search on training data
    grid.fit(X_train, y_train)
    from mpl_toolkits.mplot3d import Axes3D

    # Extract values for 3D plotting
    results_df = pd.DataFrame(grid.cv_results_)
    C_vals = results_df['param_C'].astype(float)
    epsilon_vals = results_df['param_epsilon'].astype(float)
    val_scores = results_df['mean_test_score']
    train_scores = results_df['mean_train_score'] 

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for training scores
    train_scatter = ax.scatter(C_vals, epsilon_vals, train_scores, c=train_scores, cmap='Blues', label="Training Score", s=50)

    # Scatter plot for validation scores
    val_scatter = ax.scatter(C_vals, epsilon_vals, val_scores, c=val_scores, cmap='Oranges', label="Validation Score", s=50)

    # Add colorbars
    fig.colorbar(train_scatter, ax=ax, label="Training Score", shrink=0.6, pad=0.1)
    fig.colorbar(val_scatter, ax=ax, label="Validation Score", shrink=0.6, pad=0.2)

    # Axis labels and title
    ax.set_xlabel("C")
    ax.set_ylabel("Epsilon")
    ax.set_zlabel("Score")
    ax.set_title("Training vs Validation Scores (3D Grid Search)")

    # Add a legend
    ax.legend(loc="upper left")

    # Show plot
    plt.show()
    print("Best parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

    # train SVR based on params from best CV results
    best_model = SVR(**grid.best_params_) 
    best_model.fit(X_train, y_train) 
    
    #Best parameters: {'C': 1, 'epsilon': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
    #Best score: 0.376645109255676
    # TODO: train this model on all the train data with the best params again and evalaute on test
    return best_model


def custom_importance_getter(estimator, X, y):
    result = permutation_importance(estimator, X, y, n_repeats=5, random_state=42)
    return result.importances_mean  # Mean importance of each feature

def custom_rfe_with_plot(X, y, model, n_splits=5):
    n_features = X.shape[1]
    feature_indices = np.arange(n_features)  # Keep track of feature indices
    feature_rankings = np.zeros(n_features)  # To store rankings for features
    performance_scores = []  # To store average performance at each iteration
    remaining_features = []  # Track number of features at each step
    y_binned = bin_data(y)
    # breakpoint()
    while len(feature_indices) > 1:
        print(f"Running with {len(feature_indices)} features...")
        fold_weights = np.zeros((n_splits, len(feature_indices)))
        fold_scores = []  # Track scores for current feature subset
        
        stratified_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=6)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=6)
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        # for i, (train_idx, val_idx) in enumerate(stratified_cv.split(X, y_binned)):
            # Train and evaluate model on current subset of features
            X_train, X_val = X.iloc[train_idx, feature_indices], X.iloc[val_idx, feature_indices]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train, y_train)
            score = evaluate(model, X_val, y_val)
            fold_scores.append(score)  # Calculate and save R²

            # Capture feature weights
            if hasattr(model, "coef_"):
                fold_weights[i] = np.abs(model.coef_).flatten()
            else:
                fold_weights[i] = np.abs(model.dual_coef_).mean(axis=0)

        # Average the feature weights and performance scores across folds
        avg_weights = fold_weights.mean(axis=0)
        avg_r2_score = np.mean(fold_scores)

        # Track performance for this subset of features
        performance_scores.append(avg_r2_score)
        remaining_features.append(len(feature_indices))

        # Identify and remove the least important feature
        least_important_index = np.argmin(avg_weights)
        #print(f"Removing feature {feature_indices[least_important_index]} with weight {avg_weights[least_important_index]}")
        # assign ranking based on what number it was removed - higher ranking --> removed earlier
        feature_rankings[feature_indices[least_important_index]] = len(feature_indices)
        feature_indices = np.delete(feature_indices, least_important_index)
       

    # Final feature gets rank 1
    feature_rankings[feature_indices[0]] = 1
    performance_scores.append(avg_r2_score)
    remaining_features.append(len(feature_indices))
    best_perf = max(performance_scores)
    n_features_to_select = remaining_features[performance_scores.index(best_perf)]
    print(f"Peak performance of {best_perf} at: {n_features_to_select}" )

    # Plot performance vs. number of features
    plt.figure(figsize=(10, 6))
    plt.plot(remaining_features, performance_scores, marker='o')
    plt.gca().invert_xaxis()  # Invert x-axis to show fewer features on the right
    plt.title("Model Performance vs. Number of Features")
    plt.xlabel("Number of Features")
    plt.ylabel("Average R² Score")
    plt.grid()
    plt.show()

    # sort features and take the top N feaures to return as the new DF
    index_to_rank = np.argsort(feature_rankings)
    X_ranked = X.columns[index_to_rank]
    breakpoint()
    return X_ranked, n_features_to_select
  
def svr_rfe(X, y):
    y_binned = bin_data(y)
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y_binned)
    reg = SVR(kernel='linear') 
    rfecv = RFECV(
    estimator=reg,
    step=1,
    cv=10,
    scoring="r2",
    )
    model = rfecv.fit(X, y)
    print(f"Optimal number of features: {rfecv.n_features_}")

    cv_results = pd.DataFrame(rfecv.cv_results_)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test r2_score")
    plt.errorbar(
        x=cv_results["n_features"],
        y=cv_results["mean_test_score"],
        yerr=cv_results["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()
    
    return model

def train_ridge(X_train, y_train):
     # Hyperparameter tuning for alpha using GridSearchCV
    param_grid = {'alpha': np.arange(10, 1001, 10)} # Search for alpha in log scale

    # Cross-validation grid search for optimal alpha
    grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2', return_train_score=True)
    grid_search.fit(X_train, y_train)

    train_scores = grid_search.cv_results_['mean_train_score']
    val_scores = grid_search.cv_results_['mean_test_score']
    alpha_values = grid_search.cv_results_['param_alpha'].data

    # Plot training and validation R^2 scores
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, train_scores, label='Training R^2', marker='o')
    plt.plot(alpha_values, val_scores, label='Validation R^2', marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('R² Score')
    plt.title('Training and Validation R² Scores vs Alpha')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Best alpha found from GridSearchCV
    best_alpha = grid_search.best_params_['alpha']
    print(f"Best alpha value: {best_alpha}")

    # best_alpha = 175
    # Retrain Ridge with the best alpha and evaluate again
    best_ridge_model = Ridge(alpha=best_alpha)
    best_ridge_model.fit(X_train, y_train)
    breakpoint()
    return best_ridge_model

def main():
    # inputs:
    subjects, cog_type = parse()
    X_train, X_test, y_train, y_test = data_preproc(subjects, cog_type)

    plt.hist(y_train, bins=10, alpha=0.5, label='Train')
    plt.hist(y_test, bins=10, alpha=0.5, label='Test')
    plt.legend()
    plt.show()

    # features = train_lasso(X_train, y_train)

    # try auto encoders, SVD, clustering

    # only use selected features to train svr
    # X_train = X_train
    # model = train_svr(X_train, y_train)
    # model = svr_rfe(X_train, y_train)
    model = SVR(kernel = 'linear')
    ranked_X, top_n= custom_rfe_with_plot(X_train, y_train,model)
    breakpoint()
    X_train_selected = X_train[ranked_X[:top_n]]
    # model = train_svr(X_train_selected, y_train)
    X_test_selected = X_test[ranked_X[:top_n]]
    model = train_ridge(X_train_selected, y_train)
    evaluate(model, X_test_selected, y_test)
    breakpoint()

    y_pred = model.predict(X_test_selected)
    r2_score = metrics.r2_score(y_test, y_pred)
    # print(f"R^2 score on Test Data (after tuning): {test_r2_best}")
    print("R^2 score of model", r2_score)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual")
    plt.show()
    
    
    
   


if __name__ == "__main__":
    main()
