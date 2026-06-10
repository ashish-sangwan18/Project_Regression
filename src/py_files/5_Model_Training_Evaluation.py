import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.api import OLS
from statsmodels.api import add_constant
import pickle

import os
import logging
import yaml
from dvclive import Live


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

def model_training(df, logger):
    """ Model Training """
    
    try:
        logger.debug('Model training has started')
        x_train = df.iloc[:, 1:]
        y_train = df.iloc[:, :1]

        # Adding constant to x_train and x_test
        x_train_constant = add_constant(x_train)

        model = OLS(endog = y_train, exog = x_train_constant).fit()

        # with open('./models/Regression.pkl', 'wb') as file:
        #     pickle.dump(model, file)

        logger.debug('Model has successfully trained')

        return model
    
    except Exception as e:
        logger.error('Unexpected error occurred while model training: %s', e)
        raise

def model_prediction(df, model,logger):
    """ Model Prediction """
    
    try:
        logger.debug('Model prediction has started')
        x_test = df.iloc[:, 1:]
   
        # Adding constant to x_train and x_test
        x_test_constant = add_constant(x_test)
        prediction = model.predict(x_test_constant).tolist()

        logger.debug('Model has successfully predicted')

        return prediction
    
    except Exception as e:
        logger.error('Unexpected error occurred while prediction: %s', e)
        raise

def model_accuracy(df, prediction,logger):
    """ Model Accuracy """
    
    try:
        logger.debug('Model accuracy preparation has started')

        y_test = df.iloc[:,:1]
        mape = MAPE(y_test['Wage'].tolist(), prediction)
        final_mape = np.round(mape,2)

        logger.debug('Model accuracy preparation has completed ')

        return final_mape
    
    except Exception as e:
        logger.error('Unexpected error occurred while preparation accuracy: %s', e)
        raise


def main():
    """Main Function"""

    try:
        logger_name = '5_Model_Training_Evaluation'
        logger = create_logger_handler('./logs', logger_name)

        params = load_params('./params.yaml', logger)
        test_param = params['Model_Training_Evaluation']['test_np_round']

        reading_path = './data/Train_Test/Train_Data.csv'
        df_train = load_data(reading_path, logger)

        reading_path = './data/Train_Test/Test_Data.csv'
        df_test = load_data(reading_path, logger)


        model = model_training(df_train, logger)
        prediction = model_prediction(df_test, model, logger)
        accuracy = model_accuracy(df_test, prediction, logger)


        with Live(save_dvc_exp=True) as live:
            live.log_metric('MAPE', accuracy)
            live.log_param('params', params)

        logger.debug('Model accuracy preparation has completed - %s', accuracy)
        logger.debug('Model accuracy preparation has completed - %s', test_param)

    except Exception as e:
        logger.error('Failed to complete data ingestion process - %s', e)


if __name__ == '__main__':
    main()