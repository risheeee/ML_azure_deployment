from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import os
import sys
from log_config import logging
from exception import CustomException
from dataclasses import dataclass
from utils import evaluate_models
from utils import save_obj

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model = {
                'Linear regression' : LinearRegression(),
                'ada boost' : AdaBoostRegressor(),
                'Random forest' : RandomForestRegressor(),
                'xg boost' : XGBRegressor(),
                'Decision tree' : DecisionTreeRegressor()
            }

            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = model)
            best_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]
            best_model = model[best_model_name]

            if best_score < 0.7:
                raise CustomException("no model found")
            logging.info("best model found")

            save_obj(file_path = self.model_trainer_config.trained_model_file_path,
                     obj = best_model)
            
            predict = best_model.predict(X_test)
            best_r2 = r2_score(y_test, predict)
            return best_r2
        
        except Exception as e:
            raise CustomException(e, sys)

