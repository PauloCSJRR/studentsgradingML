import os
import sys
from dataclasses import dataclass # type: ignore

from catboost import CatBoostRegreessor
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test import data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], #X_train
                train_array[:,-1],  #y_train

                test_array[:,:-1],  #X_test
                test_array[:,-1],   #y_test
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGBoost Regressor': XGBRegressor(),
                'CatBoost Regressor': CatBoostRegreessor(verbose=False),
                'AdaBoost Regressor': AdaBoostClassifier()
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            ## Get the best model from dict

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info(f'Best model found both on training and testing datasets')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
                )
                 

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square


        except Exception as e:
            raise CustomException(e, sys)