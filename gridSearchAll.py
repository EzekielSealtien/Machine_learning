import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

# Load Signatures / Feature vector
load_signatures = np.load('iris_feat_concat.npy')
# Split inputs / outputs
X = load_signatures[ : , : -1].astype('float')
Y = load_signatures[ : , -1].astype('int')

# Define the parameters grids
param_grid_svc = {
    'svc__C': [0.1, 1, 3, 10],
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__gamma': ['scale', 'auto']
}

param_grid_knn = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17],
    'knn__weights' : ['uniform', 'distance']
}

param_grid_lda = {
    'lda__solver': ['lsqr'],
    'lda__shrinkage': [None, 'auto', 0.1, 0.5]
}

# Create pipelines
pipeline_svc = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('svc', SVC())
    ]
)
pipeline_knn = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
        
    ]
)
pipeline_lda = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ]
)

# Create the GridSearchCV objects
grid_search_svc = GridSearchCV(pipeline_svc, param_grid_svc, cv=3, n_jobs=1)
grid_search_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=3, n_jobs=1)
grid_search_lda = GridSearchCV(pipeline_lda, param_grid_lda, cv=3, n_jobs=1)


# Define test proportion
train_proportion = 0.15
seed = 10
# Split train / test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=train_proportion, random_state=seed)
# Fit Models
grid_search_svc.fit(X_train, Y_train)
grid_search_knn.fit(X_train, Y_train)
grid_search_lda.fit(X_train, Y_train)

# Print the best parameters and scores
print(f'Best parameters for SVC: {grid_search_svc.best_params_}')
print(f'Best cross validation score for SVC: {grid_search_svc.best_score_}')

print(f'Best parameters for KNN: {grid_search_knn.best_params_}')
print(f'Best cross validation score for KNN: {grid_search_knn.best_score_}')

print(f'Best parameters for LDA: {grid_search_lda.best_params_}')
print(f'Best cross validation score for LDA: {grid_search_lda.best_score_}')
