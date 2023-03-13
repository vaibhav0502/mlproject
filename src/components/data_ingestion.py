import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation, DataTransformationConfig

ARTIFACTS = 'artifacts'
DATA_PATH = os.path.normpath('notebook/data/stud.csv')

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(ARTIFACTS, 'train.csv')
    test_data_path: str = os.path.join(ARTIFACTS, 'test.csv')
    raw_data_path: str = os.path.join(ARTIFACTS, 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

    def initiate_data_ingestion(self):
        '''
        Function to initiate data
        
        Returns:
                tuple(str, str): train and test data path
        
        '''

        logging.info("Data ingestion started")
        try:

            df = pd.read_csv(DATA_PATH)
            logging.info("Read dataset done.")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        except Exception as e:
            logging.error(CustomException(e, sys))
            # raise CustomException(e, sys)
            return (None, None)

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(
        train_data, test_data)
