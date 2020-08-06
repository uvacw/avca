
import pandas as pd
import os
import time
import io
import http.client, urllib.request, urllib.parse, urllib.error, base64, json
import time 
import requests
import operator
import numpy as np


def run_microsoft_classifier(microsoft_api_key, path_save, path_source, image_list):
    print('Loading Microsoft classifier. Please note this is slower due to Microsoft rate limitations (20 images per minute)')


    uri_base = 'https://westus.api.cognitive.microsoft.com'

    headers = {
        # Request headers.
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': microsoft_api_key,
    }

    params = urllib.parse.urlencode({
        # Request parameters. All of them are optional.
        'visualFeatures': 'Categories,Tags,Faces',
        'language': 'en',
    })


    filenames = os.listdir(path_source)

    filenames = [item for item in filenames if ('.jpg' in item.lower()) or ('.png' in item.lower())]
    filenames = [item for item in filenames if item.split('.')[0] in image_list]


    photos_tagged = [item.replace('.pkl','') for item in os.listdir(path_save)]


    counter = 0
    for unique_photo_id in filenames:
        results_tmp = []
        if unique_photo_id not in photos_tagged:
            result ={}
            result['unique_photo_id'] = unique_photo_id
            print('getting', unique_photo_id)
            try:
                with open( path_source+'/'+unique_photo_id, 'rb' ) as f:
                    data = f.read()

                conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
                conn.request("POST", "/vision/v1.0/analyze?%s" % params, data, headers)
                response = conn.getresponse()
                res = response.read()

                
                # print(res)

                result['tagging'] = res
                


                
            except Exception as e:
                result['tagging'] = str(e)




            time.sleep(3)

            pd.DataFrame([result]).to_pickle(path_save+'/'+unique_photo_id+'.pkl')
            
            counter += 1
            print(counter, 'out of', len(filenames), 'completed for Microsoft')


            # if counter == 20:
            #     microsoft_results.to_pickle(path_save+'/microsoft_results.pkl')
            #     microsoft_results.to_csv(path_save+'/microsoft_results.csv')

            #     print('completed batch of 20, waiting')
            #     time.sleep(58)
            #     counter = 0
        else:
            print(unique_photo_id, 'already tagged for Microsoft')


