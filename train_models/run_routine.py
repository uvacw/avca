import sys

def run_routine():
    print('Executing the first step')

    print('Loading the API keys')
    #1. Run the images through the APIs

    # Locating API keys
    from keys import api_keys
    for key in api_keys.keys():
        if api_keys[key] == None:
            raise Exception(str('API key information for ' + key + ' not filled out. \nPlease update the keys.py file.'))

    print('API keys loaded:', ' '.join(list(api_keys.keys())))

    # Running clarifai

    from helpers.models.clarifai_classifier import run_clarifai_classifier

    run_clarifai_classifier(api_keys['clarifai_api_key'], 
        'working_data/machinecoding/clarifai',
        'source/subsample')

    from helpers.models.clarifai_parser import run_clarifai_parser

    run_clarifai_parser('working_data/machinecoding/', 'working_data/machinecoding/clarifai')



    # Running Microsoft
    from helpers.models.microsoft_classifier import run_microsoft_classifier
    run_microsoft_classifier(api_keys['microsoft_api_key'], 
        'working_data/machinecoding/microsoft',
        'source/subsample')

    from helpers.models.microsoft_parser import run_microsoft_parser

    run_microsoft_parser('working_data/machinecoding/', 'working_data/machinecoding/microsoft')


    # from helpers.models.google_classifier import run_google_classifier
    run_google_classifier('working_data/machinecoding/google',
        'source/subsample')

    from helpers.models.google_parser import run_google_parser

    run_google_parser('working_data/machinecoding/', 'working_data/machinecoding/google')



    #2. Do the train/test split

    from helpers.analysis.train_test_split import run_train_test_split
    variables = run_train_test_split(train_test_ratio = 0.9)


    # #3. Do the API training and report results

    from helpers.custommodels.clarifai_trainer import create_clarifai_model, add_images_clarifai, train_clarifai_model, predict_testset_clarifai

    create_clarifai_model(api_keys['clarifai_api_key'], api_keys['clarifai_custom_model_name'], variables)

    add_images_clarifai(api_keys['clarifai_api_key'])

    model = train_clarifai_model(api_keys['clarifai_api_key'], api_keys['clarifai_custom_model_name'])

    predict_testset_clarifai(api_keys['clarifai_api_key'], api_keys['clarifai_custom_model_name'], confidence=0.5)


    from helpers.analysis.classification_report import calculate_precision_recall

    calculate_precision_recall('clarifai_custom_model.pkl', ['gen_people', 'gen_profit', 'gen_planet'], 'clarifaicustom')

    # # #4. Export tags for Expert training

    from helpers.analysis.expert_coding import export_tags_expert_coding

    export_tags_expert_coding(['gen_people', 'gen_profit', 'gen_planet'], confidence=[0.7, 0.9, 0.95, 0.99])


    # # #5. Run supervised machine learning
    from helpers.analysis.supervised_models import create_features, run_supervised_models
    from supervised_models_config import models_supervised
    # setting the confidence to create the binary datasets as 0, meaning that if a tag has a likelihood above 0 (i.e., it is detected
    # even if marginally), it will be considered as present (for the binary datasets)
    create_features(confidence_for_binary = 0)
    run_supervised_models(models_supervised, variables=['gen_people', 'gen_profit', 'gen_planet'])

    # # #6. Retrieve expert coding and run predictions
    from helpers.analysis.expert_coding import run_expert_models
    from helpers.analysis.classification_report import calculate_precision_recall
    run_expert_models(['gen_people', 'gen_profit', 'gen_planet'])

    # # #7. Unsupervised
    from helpers.analysis.unsupervised_models import create_features_unsupervised, create_unsupervised_LDA_models, select_unsupervised_models
    from unsupervised_models_config import models_unsupervised
    create_features_unsupervised(likelihoods=api_keys['unsupervised_likelihoods'], freqthresholds=api_keys['unsupervised_thresholds'])
    create_unsupervised_LDA_models(num_topics_list=api_keys['unsupervised_num_topics_list'], alphas=api_keys['unsupervised_alphas'] )
    select_unsupervised_models(['gen_people', 'gen_profit', 'gen_planet'], models_unsupervised, multiclass=False)





if __name__ == '__main__':
    run_routine()
