import os
import sys
from src.logger import logging 
from src.exception import CustomException
from src.utils import evaluate_model, save_object
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, Y_train, X_test, Y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])

            models = {
                "RandomForestClassifier": RandomForestClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "LogisticRegression": LogisticRegression(),
                "RidgeClassifier": RidgeClassifier(),
                "SVC": SVC(),
                "XGBClassifier": XGBClassifier(),
            }
            params = {
                "RandomForestClassifier": {'n_estimators': [8,16,32,64,128,256]},
                "KNeighborsClassifier": {'n_neighbors': np.arange(5,30, 2), 'weights':['uniform','distance']},
                "DecisionTreeClassifier": {'criterion':['squared_error','friedman_mse', 'absolute_error', 'poisson']},
                "AdaBoostClassifier": {'learning_rate':[0.1, 0.01, 0.5, 0.001], 'n_estimators': [8,16,32,64,128 ]},
                "LogisticRegression": {},
                "RidgeClassifier": {},
                "SVC": {'C': [0.1,1,10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': [0.1,1,'scale','auto']},
                "XGBClassifier": {'min_child_weight': [1,5,10], 'gamma': [0.5,1,1.5,2,5], 
                                  'subsample': [0.6,0.8,1.0],'colsample_bytree': [0.6, 0.8, 1.0],'max_depth': [3, 4, 5]}
            }

            report = evaluate_model(X_train, Y_train, X_test, Y_test, models, params)
            
            best_model_score = max(sorted(report.values()))
            best_model_name = list(report.keys())[list(report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info("Best model found on testing data")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            predicted = best_model.predict(X_test)
            
            acc = accuracy_score(Y_test, predicted)

            return acc

        except Exception as e:
            raise CustomException(e,sys) 