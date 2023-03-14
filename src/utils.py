import os
import sys
import time
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        ss = time.time()
        for i in range(len(list(models))):
            s = time.time()
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            logging.info(
                f"Model training started for: {list(models.keys())[i]}")

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            e = time.time()
            logging.info(
                f"Model training completed for: {list(models.keys())[i]}, time: {e-s} and R2 score, train: {train_model_score}, test: {test_model_score}")
            logging.info("--"*25)
        tt = time.time()
        logging.info(f"Total time: {tt-ss}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
