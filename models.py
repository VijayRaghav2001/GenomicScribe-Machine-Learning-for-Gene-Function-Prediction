import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

def get_model(model_name, params=None):
    """Get a machine learning model by name"""
    models = {
        'random_forest': RandomForestClassifier,
        'svm': SVC,
        'logistic_regression': LogisticRegression,
        'xgboost': xgb.XGBClassifier
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(models.keys())}")
    
    if params:
        return models[model_name](**params)
    else:
        return models[model_name]()

def train_model(model, X_train, y_train):
    """Train a machine learning model"""
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(model_name, X_train, y_train, param_grid, cv=5):
    """Perform hyperparameter tuning for a model"""
    model = get_model(model_name)
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_
