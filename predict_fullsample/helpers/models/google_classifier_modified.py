
import pandas as pd
import os
import time
import io

from google.cloud import vision
from google.cloud.vision import types


def run_google_classifier(path_save, path_source, image_list ):
	print('Loading Google classifier')
	client = vision.ImageAnnotatorClient()



	filenames = os.listdir(path_source)

	filenames = [item for item in filenames if ('.jpg' in item.lower()) or ('.png' in item.lower())]
	filenames = [item for item in filenames if item.split('.')[0] in image_list]


	photos_tagged = [item.replace('.pkl','') for item in os.listdir(path_save)]

	results = []

	counter = 0
	for unique_photo_id in filenames:
		results_tmp = []
		if unique_photo_id not in photos_tagged:
			result ={}
			result['unique_photo_id'] = unique_photo_id
			print('getting', unique_photo_id)
			try:
				with io.open(path_source+'/'+unique_photo_id, 'rb') as image_file:
				    content = image_file.read()

				image = types.Image(content=content)

				result['label_detection'] = client.label_detection(image=image)
				result['face_detection']  = client.face_detection(image=image)
				result['logo_detections'] = client.logo_detection(image=image)


				
			except Exception as e:
				result['tagging'] = str(e)




			time.sleep(0.01)

			# print(result)
			pd.DataFrame([result],).to_pickle(path_save+'/'+unique_photo_id+'.pkl')


			
		else:
			print(unique_photo_id, 'already tagged for Google')

		counter += 1
		print(counter, 'out of', len(filenames), 'completed for Google')



