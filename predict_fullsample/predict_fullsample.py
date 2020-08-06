import sys
import os
import pandas as pd


fullsample = pd.read_pickle('all_images_collected_and_in_scope.pkl')

# def correct_path(folder):
#     return folder.replace('../Data/', '/Users/theo/Dropbox/MachineVision_CorpCom/Data/')

# # fullsample['folder'] = fullsample['folder'].apply(correct_path)
# fullsample['path'] = fullsample['folder'] +  fullsample['company']

# def fix_path(path):
#     return path.replace(' ', '').replace('é', 'e')

# def fix_image_name(uniqueID):
#     return uniqueID.replace('é', 'e')

# fullsample['uniqueID'] = fullsample['uniqueID'].apply(fix_image_name)

# fullsample['path'] = fullsample['path'].apply(fix_path)

# total_images = 0
# for folder in fullsample['path'].unique().tolist():
#     images = os.listdir(folder)
#     images = [item.split('.')[0] for item in images]
#     images_data = fullsample[fullsample['path']==folder]['uniqueID']
#     images_found = [item for item in images_data if item in images]
#     print(folder, len(images_data), len(images_found))
#     total_images += len(images_found)

# print('total images:', total_images)
     


# print('Executing the first step')

# print('Loading the API keys')
# #1. Run the images through the APIs

# # Locating API keys
# from keys import api_keys
# for key in api_keys.keys():
#     if api_keys[key] == None:
#         raise Exception(str('API key information for ' + key + ' not filled out. \nPlease update the keys.py file.'))

# print('API keys loaded:', ' '.join(list(api_keys.keys())))

# Running Clarifai

# from helpers.models.clarifai_classifier_modified import run_clarifai_classifier

# for path in fullsample['path'].unique().tolist():
#     print(path)
#     image_list = fullsample[fullsample['path']==path]['uniqueID'].values.tolist()
#     run_clarifai_classifier(api_keys['clarifai_api_key'], 
#       'working_data/machinecoding/clarifai', path, image_list)

# from helpers.models.clarifai_parser import run_clarifai_parser

#run_clarifai_parser('working_data/machinecoding/', 'working_data/machinecoding/clarifai')



# # # Running Microsoft

# from helpers.models.microsoft_classifier_modified import run_microsoft_classifier

# for path in fullsample['path'].unique().tolist():
#     print(path)
#     image_list = fullsample[fullsample['path']==path]['uniqueID'].values.tolist()
#     run_microsoft_classifier(api_keys['microsoft_api_key'], 
#       'working_data/machinecoding/microsoft', path, image_list)


# from helpers.models.microsoft_parser import run_microsoft_parser

# run_microsoft_parser('working_data/machinecoding/', 'working_data/machinecoding/microsoft')


# # # Running Google

# from helpers.models.google_classifier_modified import run_google_classifier
# for path in fullsample['path'].unique().tolist():
#     print(path)
#     image_list = fullsample[fullsample['path']==path]['uniqueID'].values.tolist()
#     run_google_classifier('working_data/machinecoding/google', path, image_list)

# from helpers.models.google_parser import run_google_parser

# run_google_parser('working_data/machinecoding/', 'working_data/machinecoding/google')

# from helpers.analysis.train_test_split import run_train_test_split
# run_train_test_split(train_test_ratio = 1.0)

# from helpers.analysis.supervised_models import create_features
# create_features(confidence_for_binary = 0)


# from helpers.analysis.supervised_models import predict_cases

# # PLANET
# dataset_name = 'all_features_likelihood'
# train_set = pd.read_pickle('working_data/machinecoding/all_features_likelihood.pkl')
# classifier_name = 'supervised_SGD_gen_planet_likelihood_all_f1_weighted'
# variable = 'gen_planet'

# predict_cases(dataset_name, variable, train_set, classifier_name)


# # PEOPLE
# dataset_name = 'clarifai_features_likelihood'
# train_set = pd.read_pickle('working_data/machinecoding/clarifai_features_likelihood.pkl')
# classifier_name = 'supervised_SVC_gen_people_likelihood_clarifai_f1_weighted'
# variable = 'gen_people'

# predict_cases(dataset_name, variable, train_set, classifier_name)


# # PROFIT
# dataset_name = 'all_features_binary'
# train_set = pd.read_pickle('working_data/machinecoding/all_features_binary.pkl')
# classifier_name = 'supervised_SGD_gen_profit_binary_all_f1_weighted'
# variable = 'gen_profit'

# predict_cases(dataset_name, variable, train_set, classifier_name)

