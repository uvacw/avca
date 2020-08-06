import pandas as pd 
import os


def run_clarifai_parser(path_save, path_source):

    print('Consolidating results of Clarifai classifier')

    files = [item for item in os.listdir(path_source) if '.pkl' in item]
    print(len(files), 'to consolidate')

    df = pd.DataFrame()
    for file in files:
        try:
            tmpdf = pd.read_pickle(path_source+'/'+file)
            df = df.append(tmpdf)
        except:
            print(file, 'error')







    def parse_tagging(tagging, unique_photo_id):
        results = []

        concepts = tagging['outputs'][0]['data']['concepts']
        for concept in concepts:
            concept['unique_photo_id'] = unique_photo_id
            results.append(concept)


        return results


    results = pd.DataFrame()
    counter = 0
    for tagging, unique_photo_id in df[['tagging', 'unique_photo_id']].values.tolist():

        try:
            res = parse_tagging(tagging, unique_photo_id)
            res = pd.DataFrame(res)
            
        except:
            res = pd.DataFrame([{'unique_photo_id': unique_photo_id, 'error': 'error'}])
        results = results.append(res)
        counter += 1
        print('added', counter, 'to Clarifai results')

    results = results.rename(columns={'id': 'clarifai_concept_id', 'name': 'clarifai_label', 'value': 'clarifai_likelihood_value'})
    results['classifier'] = 'clarifai'

    results.to_pickle(path_save+'clarifai_parsed.pkl')
    results.to_csv(path_save+'clarifai_parsed.csv')
    results.to_excel(path_save+'clarifai_parsed.xlsx', index=False)
