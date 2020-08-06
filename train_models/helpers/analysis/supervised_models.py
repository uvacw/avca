import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib
from helpers.analysis.classification_report import calculate_precision_recall
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


import logging
import warnings

from logging.handlers import RotatingFileHandler

logger_file_handler = RotatingFileHandler(u'working_data/logs/supervised_models.log')
logger_file_handler.setLevel(logging.INFO)

logging.captureWarnings(True)

logger = logging.getLogger(__name__)
warnings_logger = logging.getLogger("py.warnings")

logger.addHandler(logger_file_handler)
logger.setLevel(logging.DEBUG)
warnings_logger.addHandler(logger_file_handler)



def logmsg(msg):
    msg = str(msg)
    print(msg)
    logger.info(msg)
    return

def renamecols(df, dfname):
    dfcols = {}
    for column in df.columns:
        if column != 'unique_photo_id':
            dfcols[column] = dfname+'_'+column
        
    df = df.rename(columns=dfcols)
    return df
    


def cleanuniqueid(unique_photo_id):
    return unique_photo_id.split('.')[0].replace(' ','')

def dummify(col, confidence = 0):
    if col > confidence:
        return 1
    else:
        return 0

def renametags(tag, dfname):
    return   dfname + '_' + str(tag) 


def create_features(confidence_for_binary = 0):
    logmsg('Starting with supervised machine learning')
    logmsg('Creating features dataset')

    traintest = pd.read_pickle('working_data/manualcoding/manualcoding_traintest.pkl')
    logmsg('loaded manual coding of subsample. N = ' + str(len(traintest)))

    traintest_unique_ids =  traintest['unique_photo_id'].unique().tolist()
    logmsg('total of unique IDs in subsample. N = ' + str(len(traintest_unique_ids)))



    # Loading machine coding
    google = pd.read_pickle('working_data/machinecoding/google_parsed.pkl')
    clarifai = pd.read_pickle('working_data/machinecoding/clarifai_parsed.pkl')
    microsoft = pd.read_pickle('working_data/machinecoding/microsoft_parsed.pkl')

    google['unique_photo_id'] = google['unique_photo_id'].apply(cleanuniqueid)
    clarifai['unique_photo_id'] = clarifai['unique_photo_id'].apply(cleanuniqueid)
    microsoft['unique_photo_id'] = microsoft['unique_photo_id'].apply(cleanuniqueid)


    # Ensuring alignment between manual and machine coding

    clarifai = clarifai[clarifai['unique_photo_id'].isin(traintest_unique_ids)]
    google = google[google['unique_photo_id'].isin(traintest_unique_ids)]
    microsoft = microsoft[microsoft['unique_photo_id'].isin(traintest_unique_ids)]


    logmsg('Clarifai, Google and Microsoft datasets now aligned with train/test dataset per unique id')


    logmsg('Processing Google')
    logmsg('Classifiers available:')
    logmsg(google.classifier.value_counts())

    # Creating specific dataframes per classifier
    google_label_detection = google[google['classifier'] == 'google_label_detection'][['unique_photo_id', 'label_description', 'label_mid', 'label_score',]]
    google_face_detection = google[google['classifier'] == 'google_face_detection'][['unique_photo_id', 'faces_anger_likelihood', 
                                                                                            'faces_blurred_likelihood',
           'faces_detection_confidence', 'faces_headwear_likelihood',
           'faces_joy_likelihood', 'faces_sorrow_likelihood',
           'faces_surprise_likelihood', 'faces_under_exposed_likelihood',]]
    google_logo_detection = google[google['classifier'] == 'google_logo_detection'][['unique_photo_id', 'logo_description',
           'logo_mid', 'logo_score',]]
    google_dataframes = []

    google_logo_detection['logo_description'] = google_logo_detection['logo_description'].apply(renametags, args=('google_logo',))



    logmsg('Processing logos')
    google_logo_detection_reworked = pd.pivot_table(google_logo_detection, values='logo_score', index=['unique_photo_id', ],
                         columns=['logo_description'], aggfunc=np.sum).fillna(0).reset_index()

    google_logo_detection_reworked.head()
    google_dataframes.append(google_logo_detection_reworked)

    logmsg('Processing labels')

    google_label_detection['label_description'] = google_label_detection['label_description'].apply(renametags, args=('google_label',))
    google_label_detection_reworked = pd.pivot_table(google_label_detection, values='label_score', index=['unique_photo_id', ],
                         columns=['label_description'], aggfunc=np.sum).fillna(0).reset_index()
    google_dataframes.append(google_label_detection_reworked)


    logmsg('Processing faces')
    logmsg('...generating specific variables: total faces, average faces, max faces, min faces')
    logmsg('...these variables are required when there is more than one face detected in the image')
    logmsg('...so, for example, if there are three faces, the mean likelihood of joy would be presented')

    column_faces_google = []
    google_face_detection_totalfaces = pd.DataFrame(google_face_detection.unique_photo_id.value_counts()).reset_index().rename(columns={'unique_photo_id':'google_faces_total_faces', 'index': 'unique_photo_id'})
    google_dataframes.append(google_face_detection_totalfaces)
    column_faces_google.extend(list(google_face_detection_totalfaces.columns))

    meanface = google_face_detection.groupby('unique_photo_id').mean().reset_index()
    meanface = renamecols(meanface, 'google_mean')
    column_faces_google.extend(list(meanface.columns))

    maxface = google_face_detection.groupby('unique_photo_id').max().reset_index()
    maxface = renamecols(maxface, 'google_max')
    column_faces_google.extend(list(maxface.columns))

    minface = google_face_detection.groupby('unique_photo_id').min().reset_index()
    minface = renamecols(minface, 'google_min')
    column_faces_google.extend(list(minface.columns))

    google_dataframes.append(minface)
    google_dataframes.append(maxface)
    google_dataframes.append(meanface)



    google_features = google_dataframes[0]
    for df in google_dataframes[1:]:
        google_features = google_features.merge(df, how='outer', on='unique_photo_id').fillna(0)

    logmsg('Completed Google')

    missing = set(traintest_unique_ids) - set(traintest.merge(google_features).unique_photo_id.values.tolist())


    logmsg(str('google missing: ' + str(missing)))

    dataset_likelihood_google = traintest.merge(google_features)

    dataset_likelihood_google = dataset_likelihood_google.drop_duplicates(subset='unique_photo_id')

    dataset_likelihood_google.to_pickle('working_data/machinecoding/google_features_likelihood.pkl')

    msg = 'Google likelihood dataset created. Total N = ' + str(len(dataset_likelihood_google))
    logmsg(msg)

    msg = 'Creating binary dataset for Google considering any tags with likelihood above ' + str(confidence_for_binary) + ' as present (1)'
    logmsg(msg)

    msg = 'Note: Faces are not considered in this calculation, as they do not have likelihoods expressed in the same manner as tags'
    logmsg(msg) 

    dataset_binary_google = traintest.merge(google_features)
    dataset_binary_google = dataset_binary_google.drop_duplicates(subset='unique_photo_id')

    for column in dataset_binary_google.columns:
        if column not in traintest.columns:
            if column not in column_faces_google:
                dataset_binary_google[column] = dataset_binary_google[column].apply(dummify, confidence=confidence_for_binary,)

    dataset_binary_google = dataset_binary_google.drop_duplicates(subset='unique_photo_id')

    dataset_binary_google.to_pickle('working_data/machinecoding/google_features_binary.pkl')

    msg = 'Google binary dataset created. Total N = ' + str(len(dataset_binary_google))
    logmsg(msg)

    msg = '*' * 50
    logmsg(msg)


    # Clarifai
    msg = 'Starting with Clarifai'
    logmsg(msg)

    logmsg('Processing labels')
    clarifai_labels = clarifai[['unique_photo_id', 'clarifai_label', 'clarifai_likelihood_value']]
    clarifai_labels['clarifai_label'] = clarifai['clarifai_label'].apply(renametags, args=('clarifai_label',))
    clarifai_reworked = pd.pivot_table(clarifai_labels, values='clarifai_likelihood_value', index=['unique_photo_id', ],
                        columns=['clarifai_label'], aggfunc=np.sum).fillna(0).reset_index()

    missing = set(traintest.unique_photo_id.values.tolist()) - set(traintest.merge(clarifai_reworked).unique_photo_id.values.tolist())

    logmsg(str('clarifai missing' +  str(missing)))

    dataset_likelihood_clarifai = traintest.merge(clarifai_reworked)
    dataset_likelihood_clarifai = dataset_likelihood_clarifai.drop_duplicates(subset='unique_photo_id')

    dataset_likelihood_clarifai.to_pickle('working_data/machinecoding/clarifai_features_likelihood.pkl')

    msg = 'Clarifai likelihood dataset created. Total N = ' + str(len(dataset_likelihood_clarifai))
    logmsg(msg)

    msg = 'Creating binary dataset for Clarifai considering any tags with likelihood above ' + str(confidence_for_binary) + ' as present (1)'
    logmsg(msg)
    

    dataset_binary_clarifai = traintest.merge(clarifai_reworked)
    dataset_binary_clarifai = dataset_binary_clarifai.drop_duplicates(subset='unique_photo_id')

    for column in dataset_binary_clarifai.columns:
        if column not in traintest.columns:
            dataset_binary_clarifai[column] = dataset_binary_clarifai[column].apply(dummify, confidence=confidence_for_binary)

    dataset_binary_clarifai.to_pickle('working_data/machinecoding/clarifai_features_binary.pkl')

    msg = 'Clarifai binary dataset created. Total N = ' + str(len(dataset_binary_clarifai))
    logmsg(msg)

    msg = '*' * 50
    logmsg(msg)

    # Microsoft
    msg = 'Starting with Microsoft'
    logmsg(msg)

    logmsg(microsoft['classifier'].value_counts())
    logmsg('Processing faces')
    logmsg('.. faces for Microsoft also include age (average of all faces) and female/male faces (total for each category)')

    column_faces_microsoft = []
    microsoft_faces = microsoft[microsoft.classifier=='microsoft_faces'][['unique_photo_id','microsoft_faces_age',
       'microsoft_faces_gender']]
    
    microsoft_totalfaces = pd.DataFrame(microsoft_faces.unique_photo_id.value_counts()).reset_index().rename(columns={'unique_photo_id':'microsoft_faces_total_faces', 'index': 'unique_photo_id'})

    column_faces_microsoft.extend(microsoft_totalfaces.columns)

    microsoft_totalmale = pd.DataFrame(microsoft_faces[microsoft_faces.microsoft_faces_gender =='Male'].unique_photo_id.value_counts()).reset_index().rename(columns={'unique_photo_id':'microsoft_faces_total_male_faces', 'index': 'unique_photo_id'})
    column_faces_microsoft.extend(microsoft_totalmale.columns)

    microsoft_totalfemale = pd.DataFrame(microsoft_faces[microsoft_faces.microsoft_faces_gender =='Female'].unique_photo_id.value_counts()).reset_index().rename(columns={'unique_photo_id':'microsoft_faces_total_female_faces', 'index': 'unique_photo_id'})
    column_faces_microsoft.extend(microsoft_totalfemale.columns)


    microsoft_ages = pd.pivot_table(microsoft_faces, values='microsoft_faces_age', index=['unique_photo_id', ],
                        aggfunc=np.mean).fillna(0).reset_index().rename(columns={'microsoft_faces_age': 
                                                                                'microsoft_faces_mean_age'})
    column_faces_microsoft.extend(microsoft_ages.columns)

    logmsg('Processing tags and categories')

    microsoft_tags = microsoft[microsoft.classifier=='microsoft_tags'][['unique_photo_id','microsoft_tags_name',
        'microsoft_tags_score']]


    microsoft_tags['microsoft_tags_name'] = microsoft_tags['microsoft_tags_name'].apply(renametags, args=('microsoft_tags',))
    microsoft_tags_reworked = pd.pivot_table(microsoft_tags, values='microsoft_tags_score', index=['unique_photo_id', ],
                        columns=['microsoft_tags_name'], aggfunc=np.sum).fillna(0).reset_index()
    logger.info(microsoft_tags_reworked.head())

    microsoft_category = microsoft[microsoft.classifier=='microsoft_category'][['unique_photo_id','microsoft_category_label',
        'microsoft_category_score']]


    microsoft_category['microsoft_category_label'] = microsoft_category['microsoft_category_label'].apply(renametags, args=('microsoft_category',))
    microsoft_category_reworked = pd.pivot_table(microsoft_category, values='microsoft_category_score', index=['unique_photo_id', ],
                        columns=['microsoft_category_label'], aggfunc=np.sum).fillna(0).reset_index()

    microsoft_dataframes = [microsoft_ages, microsoft_category_reworked, microsoft_tags_reworked, microsoft_totalfaces,
                        microsoft_totalfemale, microsoft_totalmale]




    microsoft_features = microsoft_dataframes[0]
    for df in microsoft_dataframes[1:]:
        microsoft_features = microsoft_features.merge(df, how='outer', on='unique_photo_id').fillna(0)


    missing = list(set(traintest.unique_photo_id.values.tolist()) - set(traintest.merge(microsoft_features).unique_photo_id.values.tolist()))


    microsoft[microsoft.unique_photo_id.isin(missing)][['unique_photo_id','error_message']].drop_duplicates()



    logmsg('items missing for Microsoft: ' + str(len(missing)))




    dataset_likelihood_microsoft = traintest.merge(microsoft_features)
    dataset_likelihood_microsoft = dataset_likelihood_microsoft.drop_duplicates(subset='unique_photo_id')
    dataset_likelihood_microsoft.to_pickle('working_data/machinecoding/microsoft_features_likelihood.pkl')

    msg = 'Microsoft likelihood dataset created. Total N = ' + str(len(dataset_likelihood_microsoft))
    logmsg(msg)

    msg = 'Creating binary dataset for Microsoft considering any tags with likelihood above ' + str(confidence_for_binary) + ' as present (1)'
    logmsg(msg)

    logmsg('... same exception made for faces (as in Google)')
    



    dataset_binary_microsoft = traintest.merge(microsoft_features)
    dataset_binary_microsoft = dataset_binary_microsoft.drop_duplicates(subset='unique_photo_id')

    for column in dataset_binary_microsoft.columns:
        if column not in traintest.columns:
            if column not in column_faces_microsoft:
                dataset_binary_microsoft[column] = dataset_binary_microsoft[column].apply(dummify, confidence=confidence_for_binary)


    dataset_binary_microsoft.to_pickle('working_data/machinecoding/microsoft_features_binary.pkl')

    msg = 'Microsoft binary dataset created. Total N = ' + str(len(dataset_binary_microsoft))
    logmsg(msg)

    msg = '*' * 50
    logmsg(msg)


    # Full dataset
    logmsg('Creating full dataset (merging Clarifai, Google, Microsoft)')

    dataset_likelihood = dataset_likelihood_google.merge(dataset_likelihood_clarifai).merge(dataset_likelihood_microsoft)
    dataset_binary = dataset_binary_google.merge(dataset_binary_clarifai).merge(dataset_binary_microsoft)

    dataset_binary.to_pickle('working_data/machinecoding/all_features_binary.pkl')
    dataset_likelihood.to_pickle('working_data/machinecoding/all_features_likelihood.pkl')

    logmsg('Datasets combining all classifiers created. Total N likelihod = ' + str(len(dataset_likelihood)) + '; binary = ' + str(len(dataset_binary)))


    datasets_likelihood = {'likelihood_all': dataset_likelihood.fillna(0), 
                        'likelihood_clarifai':dataset_likelihood_clarifai.fillna(0), 
                        'likelihood_google':dataset_likelihood_google.fillna(0), 
                        'likelihood_microsoft':dataset_likelihood_microsoft.fillna(0)}



    datasets_binary = {'binary_all': dataset_binary.fillna(0), 
                        'binary_clarifai':dataset_binary_clarifai.fillna(0), 
                        'binary_google':dataset_binary_google.fillna(0), 
                        'binary_microsoft':dataset_binary_microsoft.fillna(0)}

    pickle.dump(datasets_likelihood, open('working_data/machinecoding/datasets_likelihood.pickle', 'wb'))
    pickle.dump(datasets_binary, open('working_data/machinecoding/datasets_binary.pickle', 'wb'))

    
    logmsg('Completed. Datasets stored as a pickle file at working_data/machinecoding')

    return



def execute_pipeline(features, variable, pipeline, tuned_parameters, scoring, train_set, test_set, classifier_name):
    logmsg('executing gridsearch')
    # GridSearch and best model selection

    clf = GridSearchCV(pipeline, tuned_parameters, scoring= scoring)

    clf.fit(train_set[features], train_set[variable])
    best_clf = clf.best_estimator_


    logmsg('best model found, saving predictions and model')
    # Saving the model
    joblib.dump(best_clf, 'working_data/supervised/models/'+classifier_name+'.joblibpickle') 
    # Making predictions
    test_set[variable+'_predicted_'+classifier_name] = best_clf.predict(test_set[features])
    test_set.to_pickle('working_data/predictions/'+classifier_name+'.pkl')

    return

def get_features(dataset_name, dataset):
    if 'google' in dataset_name:
        features = [column for column in dataset.columns if 'google_' in column]
    elif 'microsoft' in dataset_name:
        features = [column for column in dataset.columns if 'microsoft_' in column]
    elif 'clarifai' in dataset_name:
        features = [column for column in dataset.columns if 'clarifai_' in column]
    else:
        features = [column for column in dataset.columns if ('clarifai_' in column) or ('microsoft_' in column) or ('google_' in column)]

    return features

def run_supervised_models(models, variables):
    datasets_likelihood = pickle.load(open('working_data/machinecoding/datasets_likelihood.pickle', 'rb'))
    datasets_binary = pickle.load(open('working_data/machinecoding/datasets_binary.pickle', 'rb'))

    datasets = {}
    for key, value in datasets_likelihood.items():
        datasets[key] = value

    for key, value in datasets_binary.items():
        datasets[key] = value

    for model in models['models']:
        for scoring in models['scoring_parameters']:
            for dataset_name, dataset in datasets.items():
                for variable in variables:
                    train_set = dataset[dataset['traintest']=='train']
                    test_set = dataset[dataset['traintest']=='test']
                    features = get_features(dataset_name, dataset)
                    pipeline = model['pipeline']
                    tuned_parameters = model['tuned_parameters']
                    classifier_name = 'supervised_' + model['model_name'] + '_' + variable + '_' + dataset_name + '_' + scoring
                    logmsg('Starting ' + classifier_name)
                    try:
                        execute_pipeline(features, variable, pipeline, tuned_parameters, scoring, train_set, test_set, classifier_name)
                        calculate_precision_recall(classifier_name+'.pkl', [variable,], classifier_name)
                    except Exception as e:
                        logmsg('error: ' + classifier_name)
                        logmsg(str(e))
    return



                    

