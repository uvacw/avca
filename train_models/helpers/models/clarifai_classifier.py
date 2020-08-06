
import pandas as pd
import os
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

import time

def run_clarifai_classifier(clarifai_api_key, path_save, path_source):
    print('Loading Clarifai classifier')
    app = ClarifaiApp(api_key=clarifai_api_key)
    model = app.models.get('general-v1.3')


    filenames = os.listdir(path_source)

    filenames = [item for item in filenames if ('.jpg' in item.lower()) or ('.png' in item.lower())]



    photos_tagged = [item.replace('.pkl','') for item in os.listdir(path_save)]


    def status(tagging):
        status = 0
        if type(tagging) ==  dict:
            if 'status_msg' in tagging.keys():
                status = tagging['status_msg']
            if 'status_code' in tagging.keys():
                status = tagging['status_code']
        if type(tagging) == str:
            if 'throttled' in tagging:
                status = 'throttled'
            if 'failed' in tagging or 'ALL_ERROR' in tagging:
                status = 'failed'
            if 'urlopen error' in tagging:
                status = 'urlopen error'
        
        return status


    counter = 0
    for unique_photo_id in filenames:
        if unique_photo_id not in photos_tagged:
            result ={}
            result['unique_photo_id'] = unique_photo_id
            print('getting', unique_photo_id)
            try:
                img = ClImage(filename=path_source+'/'+unique_photo_id)
                tagging_results = model.predict_by_filename(path_source+'/'+unique_photo_id)
                result['tagging'] = tagging_results
                # print(result)
            except Exception as e:
                print(unique_photo_id, e)
                result['tagging'] = str(e)

            pd.DataFrame([result]).to_pickle(path_save+'/'+unique_photo_id+'.pkl')

            time.sleep(0.01)
        else:
            print(unique_photo_id, 'already tagged for Clarifai')
        counter += 1
        print(counter, 'out of', len(filenames), 'completed for Clarifai')



