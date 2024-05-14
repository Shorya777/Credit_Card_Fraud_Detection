import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, Y_train, X_test, Y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            logging.info(f"GridSearch started for {list(models.keys())[i]}")
            gs = GridSearchCV(model,para,cv= 3)
            gs.fit(X_train, Y_train)
            logging.info(f"best parameters found for model {list(models.keys())[i]}")
            model.set_params(**gs.best_params_)
            model.fit(X_train, Y_train)
            logging.info(f"training data fitted on model {list(models.keys())[i]}")

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)
            save_object(f'artifacts\{list(models.keys())[i]}_{Y_test_pred}', model)
            
            # train_model_score = accuracy_score(Y_train, Y_train_pred)
            test_model_score = accuracy_score(Y_test, Y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)