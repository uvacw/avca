import pandas as pd 
import json
import os



def run_microsoft_parser(path_save, path_source):

    print('Consolidating results of Microsoft classifier')

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

        results = pd.DataFrame()

        tagging = json.loads(tagging)



        if 'categories' in tagging.keys():

            categories = tagging['categories']
            cat_df = []
            for category in categories:
                res = {}
                res['unique_photo_id'] = unique_photo_id
                res['microsoft_category_label'] = category['name']
                res['microsoft_category_score'] = category['score']
                res['classifier'] = 'microsoft_category'
                cat_df.append(res)

            results = results.append(pd.DataFrame(cat_df))


        tags = tagging['tags']
        tags_df = []
        for tag in tags:
            res = {}
            res['unique_photo_id'] = unique_photo_id
            res['classifier'] = 'microsoft_tags'
            res['microsoft_tags_name'] = tag['name']
            res['microsoft_tags_score'] = tag['confidence']
            tags_df.append(res)

        results = results.append(pd.DataFrame(tags_df))


        faces_df = []

        for face in tagging['faces']:
            res = {}
            res['classifier'] = 'microsoft_faces'
            res['unique_photo_id'] = unique_photo_id
            res['microsoft_faces_age'] = face['age']
            res['microsoft_faces_gender'] = face['gender']
            faces_df.append(res)

        results = results.append(pd.DataFrame(faces_df))

        return results


    results = pd.DataFrame()
    counter = 0

    for tagging, unique_photo_id in df[['tagging', 'unique_photo_id']].values.tolist():
        try:
            res = parse_tagging(tagging, unique_photo_id)
            res = pd.DataFrame(res)
            results = results.append(res)
        except:
            res = {}
            res['unique_photo_id'] = unique_photo_id
            res['error_message'] = tagging
            results = results.append(pd.DataFrame([res,]))
        counter += 1
        print('added', counter, 'to Microsoft results')


    results.to_pickle(path_save+'/microsoft_parsed.pkl')
    results.to_csv(path_save+'/microsoft_parsed.csv')
    results.to_excel(path_save+'/microsoft_parsed.xlsx', index=False)
