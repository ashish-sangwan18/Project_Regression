# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

import os
import logging
import yaml


# In Python, calling logging.getLogger(name) multiple times with the exact same name will neither overwrite the logger nor create multiple copies
# ; instead, it will always return a reference to the exact same singleton object. 

# However, how you configure that logger can cause an unexpected side effect: it will create multiple copies of your log outputs (duplicated logs)
# if you keep adding handlers to it



def create_logger_handler(log_folder, logger_name):
    # Ensure log directory exists
    log_dir = log_folder
    os.makedirs(log_dir, exist_ok=True)

    # logging configuration
    logger = logging.getLogger(logger_name)
    logger.setLevel('DEBUG')


    # This is to print at console level
    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

  # This is to print in log file
    file_path = os.path.join(log_dir, logger_name+'.log')
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel('DEBUG')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def load_params(param_file ,logger):
    """ This function is to load param.yaml file """

    try:
        with open(param_file, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug('params.yaml file loaded successfully')
        return params
        
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise



def load_data(path, logger):
    """ Loading data """
    
    try:
        logger.debug('Data load has started')
        df = pd.read_csv(path)
        logger.debug('Data loaded successfully from : %s', path)
        return df
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def save_data(df, raw_path, train_path, test_path, logger, test_data_size):
    """ Saving Data """

    try:
        logger.debug('Saving raw data has started')

        df.to_csv(raw_path, index=None)

        logger.debug('Raw Data saved successfully')


        logger.debug('Saving train_test data has started')

        train_data, test_data = train_test_split(df, test_size=test_data_size, random_state=123, shuffle=True)
        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        train_data.to_csv(train_path, index=None)
        test_data.to_csv(test_path, index=None)

        logger.debug('Train_Test Data saved successfully')


    except Exception as e:
        logger.error('There was some error while saving data - ', e)
        raise



def main():

    try:
        logger_name = '1_Data_Ingestion'
        logger = create_logger_handler('./logs', logger_name)

        reading_path = 'https://github.com/ashish-sangwan18/Datasets/raw/refs/heads/main/wagedata.csv'
        df = load_data(reading_path, logger)

        raw_path = './data/raw/' + 'Raw_Data.csv'
        train_path = './data/Train_Test/' + 'Train_Data.csv'
        test_path = './data/Train_Test/' + 'Test_Data.csv'


        # test_size=0.2
        params = load_params('./params.yaml', logger)
        test_size = params['Data_Ingestion']['test_size']
        
        
        save_data(df, raw_path, train_path, test_path, logger, test_size)

    except Exception as e:
        logger.error('Failed to complete data ingestion process - %s', e)

if __name__ == '__main__':
    main()






