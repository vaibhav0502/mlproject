import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


NUMERICAL_COLUMNS = ["writing_score", "reading_score"]
CATEGORICAL_COLUMNS = [
    "gender", "race_ethnicity", "parental_level_of_education",
    "lunch", "test_preparation_course",
]
TARGET_COLUMN_NAME ="math_score"

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            This function is responsible for returning data transformation pipeline.
        '''
        try:
            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scalar", StandardScaler())
                ]
            )
            logging.info(f"Numerical columns: {NUMERICAL_COLUMNS}")

            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder", OneHotEncoder()),
                ("scalar", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {CATEGORICAL_COLUMNS}")

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, NUMERICAL_COLUMNS),
                ('cat_pipeline', cat_pipeline, CATEGORICAL_COLUMNS)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
            This function is responsible data transformation.
        '''

        try:
            logging.info(f"Data transformation started")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data completed")

            logging.info("Call data preprocessor object")
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN_NAME], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN_NAME]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN_NAME], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN_NAME]
            
            logging.info(f"Applying preprocessing object on training and testing dataframe.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            preprocessor_obj_file_path = self.data_transformation_config.preprocessor_obj_file_path
            save_object(
                file_path=preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object to path: {preprocessor_obj_file_path}.")
            logging.info(f"Data transformation completed. Returing train, test array and preprocessing object path.")

            return (
                train_arr,
                test_arr,
                preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)

