import pandas as pd
import numpy as np
import os

from helpers.analysis.classification_report import calculate_precision_recall

import logging
import warnings

from logging.handlers import RotatingFileHandler

logger_file_handler = RotatingFileHandler(u'working_data/logs/expert_coding.log')
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

def dummify(col, confidence = 0):
    if col > confidence:
        return 1
    else:
        return 0

def cleanuniqueid(unique_photo_id):
    return unique_photo_id.split('.')[0].replace(' ','')

def renametags(tag, dfname):
    return   dfname + '_' + str(tag)

def export_tags_expert_coding(variables, confidence=[0.9, 0.95]):
    msg = 'Starting with tag export for expert coding'
    print(msg)
    logger.info(msg)

    manualcoding = pd.read_excel('source/manualcoding/manual_coding.xlsx')
    manualcoding_unique_ids = manualcoding[['unique_photo_id']].drop_duplicates()
    msg = 'loaded manual coding of subsample. N = ' + str(len(manualcoding))
    print(msg)
    logger.info(msg)


    # Loading machine coding
    google = pd.read_pickle('working_data/machinecoding/google_parsed.pkl')
    clarifai = pd.read_pickle('working_data/machinecoding/clarifai_parsed.pkl')
    microsoft = pd.read_pickle('working_data/machinecoding/microsoft_parsed.pkl')

    google['unique_photo_id'] = google['unique_photo_id'].apply(cleanuniqueid)
    clarifai['unique_photo_id'] = clarifai['unique_photo_id'].apply(cleanuniqueid)
    microsoft['unique_photo_id'] = microsoft['unique_photo_id'].apply(cleanuniqueid)


    # Ensuring alignment between manual and machine coding

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


    common_dataset = microsoft_unique_ids.merge(google_unique_ids).merge(clarifai_unique_ids).merge(manualcoding_unique_ids)
    common_dataset_unique_ids = common_dataset['unique_photo_id'].values.tolist()

    msg = 'common dataset has ' + str(len(common_dataset)) + ' unique items'
    print(msg)
    logger.info(msg)


    clarifai = clarifai[clarifai['unique_photo_id'].isin(common_dataset_unique_ids)]
    google = google[google['unique_photo_id'].isin(common_dataset_unique_ids)]
    microsoft = microsoft[microsoft['unique_photo_id'].isin(common_dataset_unique_ids)]


    cats = pd.DataFrame()

    google_labels = google[google.classifier == 'google_label_detection'][['classifier','label_description', 'label_mid', 'label_score']]

    google_labels  = google_labels.rename(columns={'label_description': 'tag',
                                                 'label_mid': 'tag_id',
                                                 'label_score': 'tag_likelihood'})

    clarifai_labels = clarifai[clarifai.error.isnull()][['classifier', 'clarifai_label', 'clarifai_concept_id',
                                                        'clarifai_likelihood_value']]
    
    clarifai_labels = clarifai_labels.rename(columns={'clarifai_label': 'tag',
                                                 'clarifai_concept_id': 'tag_id',
                                                 'clarifai_likelihood_value': 'tag_likelihood'})

    microsoft_tags = microsoft[microsoft.classifier=='microsoft_tags'][['classifier', 'microsoft_tags_name',
       'microsoft_tags_score']]

    microsoft_tags = microsoft_tags.rename(columns={'microsoft_tags_name': 'tag',
                                                'microsoft_tags_score': 'tag_likelihood'
                                               })

    microsoft_cats = microsoft[microsoft.classifier=='microsoft_category'][['classifier', 'microsoft_category_label',
       'microsoft_category_score']]

    microsoft_cats = microsoft_cats.rename(columns={'microsoft_category_label': 'tag',
                                                'microsoft_category_score': 'tag_likelihood'
                                               })

    cats = pd.concat([google_labels, clarifai_labels, microsoft_tags, microsoft_cats])


    cats['unique_tag_id'] = cats['classifier'] + '_' + cats['tag']

    print(cats.head())

    mean_likelihood = pd.pivot_table(cats, values='tag_likelihood', index=['unique_tag_id', ],
                      aggfunc=np.mean).fillna(0).reset_index()

    count_tags = pd.DataFrame(cats['unique_tag_id'].value_counts()).reset_index().rename(columns={'unique_tag_id': 'freq',
                                                                                             'index': 'unique_tag_id'})


    cats_expert = cats[['classifier', 'unique_tag_id', 'tag', 'tag_id']].drop_duplicates()
    cats_expert = cats_expert.merge(mean_likelihood, how='left').rename(columns={'tag_likelihood': 'mean_likelihood'})
    cats_expert = cats_expert.merge(count_tags, how='left')


    for confidence_level in confidence:

        cats_threshold = cats[cats.tag_likelihood >= confidence_level]

        mean_likelihood_threshold = pd.pivot_table(cats_threshold, values='tag_likelihood', index=['unique_tag_id', ],
                          aggfunc=np.mean).fillna(0).reset_index()
        count_tags_threshold = pd.DataFrame(cats_threshold['unique_tag_id'].value_counts()).reset_index().rename(columns={'unique_tag_id': 'freq' + str(confidence_level),
                                                                                                 'index': 'unique_tag_id'})

        cats_expert = cats_expert.merge(mean_likelihood_threshold, how='left').rename(columns={'tag_likelihood': 'mean_likelihood'+str(confidence_level)}).fillna(0)
        cats_expert = cats_expert.merge(count_tags_threshold, how='left').fillna(0)

    cats_expert = cats_expert.sort_values(by='freq', ascending=False)
    

    for variable in variables:
        cats_expert[variable+'_expert_decision'] = None


    cats_expert.to_excel('working_data/expertcoding/tags_for_expert_coding.xlsx', index=False)

    msg = 'Exported tags for expert coding. See file working_data/expertcoding/tags_for_expert_coding.xlsx and complete columns with expert decisions (0 = not relevant, 1 = relevant for category)'
    print(msg)
    logger.info(msg)

    return

def check_tag_presence(unique_photo_id, dataset, tags_from_expert):
    tags_from_classifiers = dataset[(dataset['unique_photo_id']==unique_photo_id)]['label'].values.tolist()
    for tag in tags_from_expert:
        if tag in tags_from_classifiers:
            return 1
    return 0

def predict_expert_coding(test_set, dataset, expert_rules, variable, classifier_name):
    tags_from_expert = expert_rules[expert_rules[variable+'_expert_decision']==1]['unique_tag_id'].values.tolist()
    test_set[variable+'_predicted_'+classifier_name] = test_set['unique_photo_id'].apply(check_tag_presence, args=(dataset, tags_from_expert))
    return test_set


def run_expert_models(variables):
    logmsg('Starting with predictions for Expert coding')
    logmsg('... looking for expert coding selections at working_data/expertcoding')
    logmsg('... files should be named expert_coding_<<MODELNAME>>.xlsx')
    models = [item for item in os.listdir('working_data/expertcoding') if ('.xlsx' in item) and (item.startswith('expert_coding_'))]
    if len(models) == 0:
        logmsg('ERROR: no files with coding selections found')
        return

    logmsg('models_found: ' + ', '.join(models))

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


    # Google Labels
    google_label_detection = google[google['classifier'] == 'google_label_detection'][['unique_photo_id', 'label_description', 'label_mid', 'label_score',]]
    logmsg('Loaded Google Labels, total rows = ' + str(len(google_label_detection)))
    google_label_detection['label_description'] = google_label_detection['label_description'].apply(renametags, args=('google_label_detection_',))
    # google_label_detection_reworked = pd.pivot_table(google_label_detection, values='label_score', index=['unique_photo_id', ],
    #                  columns=['label_description'], aggfunc=np.sum).fillna(0).reset_index()

    google_label_detection = google_label_detection.rename(columns={'label_description': 'label', 'label_score': 'likelihood'})
    google_label_detection['classifier'] = 'google_label_detection'



    # Clarifai Labels
    clarifai_labels = clarifai[['unique_photo_id', 'clarifai_label', 'clarifai_likelihood_value']]
    logmsg('Loaded Clarifai Labels, total rows = ' + str(len(clarifai_labels)))

    clarifai_labels['clarifai_label'] = clarifai_labels['clarifai_label'].apply(renametags, args=('clarifai_',))
    # clarifai_labels_reworked = pd.pivot_table(clarifai_labels, values='clarifai_likelihood_value', index=['unique_photo_id', ],
    #                  columns=['clarifai_label'], aggfunc=np.sum).fillna(0).reset_index()
    clarifai_labels = clarifai_labels.rename(columns={'clarifai_label': 'label', 'clarifai_likelihood_value': 'likelihood'})
    clarifai_labels['classifier'] = 'clarifai'


    # Microsoft Labels
    microsoft_tags = microsoft[microsoft.classifier=='microsoft_tags'][['unique_photo_id','microsoft_tags_name',
        'microsoft_tags_score']]
    logmsg('Loaded Microsoft Tags, total rows = ' + str(len(microsoft_tags)))

    microsoft_tags['microsoft_tags_name'] = microsoft_tags['microsoft_tags_name'].apply(renametags, args=('microsoft_tags',))
    # microsoft_tags_reworked = pd.pivot_table(microsoft_tags, values='microsoft_tags_score', index=['unique_photo_id', ],
    #                     columns=['microsoft_tags_name'], aggfunc=np.sum).fillna(0).reset_index()
    microsoft_tags = microsoft_tags.rename(columns={'microsoft_tags_name': 'label', 'microsoft_tags_score': 'likelihood'})
    microsoft_tags['classifier'] = 'microsoft_tags'



    microsoft_category = microsoft[microsoft.classifier=='microsoft_category'][['unique_photo_id','microsoft_category_label',
    'microsoft_category_score']]
    logmsg('Loaded Microsoft Category, total rows = ' + str(len(microsoft_category)))

    microsoft_category['microsoft_category_label'] = microsoft_category['microsoft_category_label'].apply(renametags, args=('microsoft_category',))
    # microsoft_category_reworked = pd.pivot_table(microsoft_category, values='microsoft_category_score', index=['unique_photo_id', ],
    #                     columns=['microsoft_category_label'], aggfunc=np.sum).fillna(0).reset_index()
    microsoft_category = microsoft_category.rename(columns={'microsoft_category_label': 'label', 'microsoft_category_score': 'likelihood'})
    microsoft_category['classifier'] = 'microsoft_tags'

    logmsg('Creating dataset with tags for all APIs')
    # dataset = traintest.merge(clarifai_labels_reworked, how="left").merge(microsoft_tags_reworked, how="left").merge(microsoft_category_reworked, how="left").merge(google_label_detection_reworked, how="left")
    # dataset = dataset.fillna(0)
    dataset = google_label_detection.append(microsoft_category).append(microsoft_tags).append(clarifai_labels)
    dataset = dataset[['unique_photo_id', 'classifier', 'label', 'likelihood']]

    for model in models:
        model_name = model.replace('.xlsx','')
        confidence = model_name.split('_')[-1]
        confidence = confidence.replace('l','')
        confidence = float(confidence)
        expert_rules = pd.read_excel('working_data/expertcoding/'+model)
        expert_rules = expert_rules.fillna(0)
        test_set = traintest[traintest['traintest'] == 'test']

        logmsg('...total rows for labels ' + str(len(dataset)))
        logmsg('...eliminating all labels with confidence lower than ' + str(confidence))
        dataset_for_predictions = dataset[dataset['likelihood']>= confidence]
        logmsg('...total rows now = ' + str(len(dataset_for_predictions)))



        logmsg('Starting with predictions for ' + model_name)
        for variable in variables:
            classifier_name = model_name
            logmsg('...' + variable)
            test_set = predict_expert_coding(test_set, dataset_for_predictions, expert_rules, variable, classifier_name)

        test_set.to_pickle('working_data/predictions/'+model_name+'.pkl')
        calculate_precision_recall(model_name+'.pkl', variables, model_name)

    logmsg('...completed predictions based on expert coding')



    return
