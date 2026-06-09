import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import warnings
warnings.filterwarnings('ignore')

import os
import logging



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




def data_analysis(logger):

    try:
        
        logger.debug('Data Analysis is done')

    except Exception as e:
        logger.debug('Data Analysis step has failed - %s', e)
        raise


def main():

    try:
        logger_name = '2_Data_Analysis'
        logger = create_logger_handler('./logs', logger_name)

        data_analysis(logger)
    
    except Exception as e:
        logger.error("Data Analysis notebook has faile - %s", e)
        raise


if __name__ == '__main__':
    main()


