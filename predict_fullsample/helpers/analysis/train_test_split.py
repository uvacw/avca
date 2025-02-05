import pandas as pd
import numpy as np

import logging
import warnings

from logging.handlers import RotatingFileHandler

logger_file_handler = RotatingFileHandler(u'working_data/logs/train_test_split.log')
logger_file_handler.setLevel(logging.INFO)

logging.captureWarnings(True)

logger = logging.getLogger(__name__)
warnings_logger = logging.getLogger("py.warnings")

logger.addHandler(logger_file_handler)
logger.setLevel(logging.DEBUG)
warnings_logger.addHandler(logger_file_handler)


def cleanuniqueid(unique_photo_id):
    return unique_photo_id.split('.')[0].replace(' ','')

def run_train_test_split(train_test_ratio = None):
    msg = 'Starting with train/test split'
    print(msg)
    logger.info(msg)


    # Loading machine coding
    google = pd.read_pickle('working_data/machinecoding/google_parsed.pkl')
    clarifai = pd.read_pickle('working_data/machinecoding/clarifai_parsed.pkl')
    microsoft = pd.read_pickle('working_data/machinecoding/microsoft_parsed.pkl')

    google['unique_photo_id'] = google['unique_photo_id'].apply(cleanuniqueid)
    clarifai['unique_photo_id'] = clarifai['unique_photo_id'].apply(cleanuniqueid)
    microsoft['unique_photo_id'] = microsoft['unique_photo_id'].apply(cleanuniqueid)

    # Note: as comparison is not being done anymore, the dataset does not need to contain labels for all images

    google = google[google['classifier'] == 'google_label_detection']
    google_unique_ids = google[['unique_photo_id']].drop_duplicates()
    msg = 'google has ' + str(len(google_unique_ids)) + ' unique items'
    print(msg)
    logger.info(msg)

    microsoft = microsoft[microsoft['classifier'].isin(['microsoft_tags', 'microsoft_category'])]
    microsoft = microsoft[microsoft['error_message'].isna() == True]
    microsoft_unique_ids = microsoft[['unique_photo_id']].drop_duplicates()
    msg = 'microsoft has ' + str(len(microsoft_unique_ids)) + ' unique items'
    print(msg)
    logger.info(msg)


    clarifai = clarifai[clarifai['error'].isna() == True]
    clarifai_unique_ids = clarifai[['unique_photo_id']].drop_duplicates()
    msg = 'clarifai has ' + str(len(clarifai_unique_ids)) + ' unique items'
    print(msg)
    logger.info(msg)


    common_dataset = microsoft_unique_ids.merge(google_unique_ids, how='outer').merge(clarifai_unique_ids, how='outer')
    common_dataset = common_dataset[['unique_photo_id']]
    common_dataset_unique_ids = common_dataset['unique_photo_id'].values.tolist()

    msg = 'common dataset has ' + str(len(common_dataset)) + ' unique items'
    print(msg)
    logger.info(msg)


    if not train_test_ratio:
        msg = 'train_test_ratio not informed. please check the code.'
        logger.info(msg)
        raise Exception(msg)


    train=common_dataset.sample(frac=train_test_ratio,random_state=42)
    test=common_dataset.drop(train.index)

    train['traintest'] = 'train'
    test['traintest'] = 'test'

    manualcoding_traintest = train.append(test)


    msg = 'descriptives for training and test dataset\n\n' + str(manualcoding_traintest.groupby('traintest').describe().transpose())
    print(msg)
    logger.info(msg)


    manualcoding_traintest.to_pickle('working_data/fullsample/fullsample_traintest.pkl')
    manualcoding_traintest.to_excel('working_data/fullsample/fullsample_traintest.pkl.xlsx', index=False)
    manualcoding_traintest.to_csv('working_data/fullsample/fullsample_traintest.csv')

    return 
