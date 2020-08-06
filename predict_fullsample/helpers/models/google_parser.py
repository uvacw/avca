import pandas as pd 
import os


def run_google_parser(path_save, path_source):

    print('Consolidating results of Google classifier')


    files = [item for item in os.listdir(path_source) if '.pkl' in item]
    print(len(files), 'to consolidate')

    df = pd.DataFrame()
    for file in files:
        try:
            tmpdf = pd.read_pickle(path_source+'/'+file)
            df = df.append(tmpdf)
        except:
            print(file, 'error')






    def parse_tagging(face_detection, label_detection, logo_detections, unique_photo_id):
        results = pd.DataFrame()

        if label_detection:
            labels = []

            if type(label_detection) == float:
                pass
            else:
                for label in label_detection.label_annotations:
                    res = {}
                    res['label_mid'] = label.mid
                    res['label_description'] = label.description
                    res['label_score'] = label.score
                    res['unique_photo_id'] = unique_photo_id
                    res['classifier'] = 'google_label_detection'
                    labels.append(res)

            results = results.append(pd.DataFrame(labels))



        if face_detection:
            faces = []
            if type(face_detection) == float:
                pass
            else:

                for face in face_detection.face_annotations:
                    res = {}
                    res['faces_detection_confidence'] = face.detection_confidence
                    res['faces_joy_likelihood'] = face.joy_likelihood
                    res['faces_sorrow_likelihood'] = face.sorrow_likelihood
                    res['faces_anger_likelihood'] = face.anger_likelihood
                    res['faces_surprise_likelihood'] = face.surprise_likelihood
                    res['faces_under_exposed_likelihood'] = face.under_exposed_likelihood
                    res['faces_blurred_likelihood'] = face.blurred_likelihood
                    res['faces_headwear_likelihood'] = face.headwear_likelihood
                    res['unique_photo_id'] = unique_photo_id
                    res['classifier'] = 'google_face_detection'
                    faces.append(res)
                results = results.append(pd.DataFrame(faces))

        if logo_detections:
            logos = []
            if type(logo_detections) == float:
                pass
            else:
                for logo in logo_detections.logo_annotations:
                    res = {}
                    res['logo_description'] = logo.description
                    res['logo_mid'] = logo.mid
                    res['logo_score'] = logo.score
                    res['unique_photo_id'] = unique_photo_id
                    res['classifier'] = 'google_logo_detection'
                    logos.append(res)

            results = results.append(pd.DataFrame(logos))



        return results


    print(df)
    results = pd.DataFrame()
    counter = 0
    for face_detection, label_detection, logo_detections, unique_photo_id in df[['face_detection',  'label_detection', 'logo_detections','unique_photo_id']].values.tolist():
        try:
            res = parse_tagging(face_detection, label_detection, logo_detections, unique_photo_id)
            results = results.append(res)
            print('completed ', counter, 'for Google')
        except:
            print('failed', counter)
        counter += 1




    results.to_pickle(path_save+'/google_parsed.pkl')
    results.to_csv(path_save+'/google_parsed.csv')
    results.to_excel(path_save+'/google_parsed.xlsx', index=False)
