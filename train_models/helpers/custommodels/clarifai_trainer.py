
import pandas as pd
import os
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

import time

import logging
import warnings

from logging.handlers import RotatingFileHandler

logger_file_handler = RotatingFileHandler(u'working_data/logs/clarifai_custom_model.log')
logger_file_handler.setLevel(logging.INFO)

logging.captureWarnings(True)

logger = logging.getLogger(__name__)
warnings_logger = logging.getLogger("py.warnings")

logger.addHandler(logger_file_handler)
logger.setLevel(logging.DEBUG)
warnings_logger.addHandler(logger_file_handler)


def add_images_clarifai(clarifai_api_key):
    msg = 'Clarifai custom training: Uploading images'
    logger.info(msg)
    print(msg)

    app = ClarifaiApp(api_key=clarifai_api_key)

    current_images = [item.dict()['id'] for item in app.inputs.get_all()]


    msg = ', '.join(current_images) + ' already in Clarifai'

    logger.info(msg)
    print(msg)

    manualcoding = pd.read_pickle('working_data/manualcoding/manualcoding_traintest.pkl')

    train = manualcoding[manualcoding['traintest'] == 'train']


    files = os.listdir('source/subsample')

    variables = [item for item in train.columns if item not in ['unique_photo_id', 'traintest']]
    

    images = []
    counter = 0
    i = 0
    while counter < len(train):
        item = train.iloc[counter]
        filename = 'source/subsample/' + [file for file in files if item['unique_photo_id'] in file][0]
        concepts = []
        not_concepts = []

        for variable in variables:
            if item[variable] == 1:
                concepts.append(variable)
            if item[variable] == 0:
                not_concepts.append(variable)
        

        img = ClImage(filename=filename, concepts=concepts, not_concepts=not_concepts, image_id = item['unique_photo_id'])
        if item['unique_photo_id'] not in current_images:
            images.append(img)
            msg = filename  + 'concepts: ' + ', '.join(concepts) + 'not concepts: ' + ', '.join(not_concepts) + ' added to Clarifai list'
            logger.info(msg)
            print(msg)

        else:
            msg = item['unique_photo_id'] + ' already uploaded to Clarifai'
            logger.info(msg)
            print(msg)

        if i == 128:
            if len(images) > 0:
                app.inputs.bulk_create_images(images)
                msg = 'Uploaded batch to Clarifai'
                logger.info(msg)
                print(msg)
                i = 0
                images = []


        counter += 1
        i += 1

    app.inputs.bulk_create_images(images)
    msg = 'Uploaded final batch to Clarifai'
    logger.info(msg)
    print(msg)




    




    return 


def create_clarifai_model(clarifai_api_key, model_name, variables):
    msg = 'Clarifai custom training: Creating model'
    logger.info(msg)
    print(msg)

    app = ClarifaiApp(api_key=clarifai_api_key)

    # Checking if model already created:
    models = app.models.search(model_name=model_name)

    if len(models) > 0:
        msg = model_name +  ' already available'
        logger.info(msg)
        print(msg)

        return models[0]

    else:
        model = app.models.create(model_name, concepts=variables)
        print(model.dict())
        logger.info(model())

        return model


def train_clarifai_model(clarifai_api_key, model_name):
    msg = 'Clarifai custom training: Training model'
    logger.info(msg)
    print(msg)

    app = ClarifaiApp(api_key=clarifai_api_key)

    model = app.models.get(model_name=model_name)

    results = model.train()

    print(results.dict())
    logger.info(str(results.dict()))


    return model


def predict_testset_clarifai(clarifai_api_key, model_name, confidence=0.5):
    msg = 'Clarifai custom training: Getting predictions for the test set'
    logger.info(msg)
    print(msg)

    app = ClarifaiApp(api_key=clarifai_api_key)
    model = app.models.get(model_name=model_name)

    

    manualcoding = pd.read_pickle('working_data/manualcoding/manualcoding_traintest.pkl')
    test = manualcoding[manualcoding['traintest'] == 'test']

    msg = 'Number of images in the test set: ' + str(len(test))
    logger.info(msg)
    print(msg)

    files = os.listdir('source/subsample')


    results = pd.DataFrame()
    

    images = []
    counter = 0
    i = 0
    while counter < len(test):
        item = test.iloc[counter]
        filename = 'source/subsample/' + [file for file in files if item['unique_photo_id'] in file][0]

        

        img = ClImage(filename=filename)
        pred = model.predict([img])

        item['pred'] = pred

        if 'outputs' in pred.keys():
            if 'data' in pred['outputs'][0].keys():
                if 'concepts' in pred['outputs'][0]['data'].keys():
                    for concept in pred['outputs'][0]['data']['concepts']:
                        item[concept['name'] + '_predicted_clarifaicustom_likelihood'] = concept['value']
                        if concept['value'] > confidence:
                            item[concept['name'] + '_predicted_clarifaicustom'] = 1
                        else:
                            item[concept['name'] + '_predicted_clarifaicustom'] = 0

        results = results.append(pd.DataFrame([item,]))

        counter += 1
        i += 1
        msg = 'Completed ' + str(counter) + ' predictions from the test set'
        print(msg)
        logger.info(msg)

        if i == 20:
            results.to_pickle('working_data/predictions/clarifai_custom_model.pkl')
            i = 0

    msg = 'Completed predictions for test set based on Clarifai custom model'
    print(msg)
    logger.info(msg)
    results.to_pickle('working_data/predictions/clarifai_custom_model.pkl')


    return

        


        





    




     

