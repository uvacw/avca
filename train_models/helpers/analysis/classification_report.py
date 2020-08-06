from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd 


import time

import logging
import warnings

from logging.handlers import RotatingFileHandler

logger_file_handler = RotatingFileHandler(u'working_data/logs/precision_recall.log')
logger_file_handler.setLevel(logging.INFO)

logging.captureWarnings(True)

logger = logging.getLogger(__name__)
warnings_logger = logging.getLogger("py.warnings")

logger.addHandler(logger_file_handler)
logger.setLevel(logging.DEBUG)
warnings_logger.addHandler(logger_file_handler)

def parse_cl_report(cl_report):
    cl_report = cl_report.split()
    report = {}
    report['precision_0'] = cl_report[5]
    report['recall_0'] = cl_report[6]
    report['f1score_0'] = cl_report[7]
    report['support_0'] = cl_report[8]

    report['precision_1'] = cl_report[10]
    report['recall_1'] = cl_report[11]
    report['f1score_1'] = cl_report[12]
    report['support_1'] = cl_report[13]
    
    report['precision_avg'] = cl_report[17]
    report['recall_avg'] = cl_report[18]
    report['f1score_avg'] = cl_report[19]
    report['support_avg'] = cl_report[20]
    
    return report




def calculate_precision_recall(predictions_file, variables, model_name):
	try:
		results = pd.read_pickle('results/results_classifications.pkl')
	except:
		results = pd.DataFrame()


	predictions = pd.read_pickle('working_data/predictions/'+predictions_file)

	for variable in variables:
		res = {}
		res['variable'] = variable
		res['model_name'] = model_name
		res['precision_1'] = precision_score(predictions[variable], predictions[variable+'_predicted_'+model_name], average='binary', pos_label=1)
		res['precision_0'] =  precision_score(predictions[variable], predictions[variable+'_predicted_'+model_name], average='binary', pos_label=0)
		res['precision_avg'] = precision_score(predictions[variable], predictions[variable+'_predicted_'+model_name], average='weighted')

		res['recall_1'] = recall_score(predictions[variable], predictions[variable+'_predicted_'+model_name], average='binary', pos_label=1)
		res['recall_0'] =  recall_score(predictions[variable], predictions[variable+'_predicted_'+model_name], average='binary', pos_label=0)
		res['recall_avg'] = recall_score(predictions[variable], predictions[variable+'_predicted_'+model_name], average='weighted')

		res['f1_score_1'] = f1_score(predictions[variable], predictions[variable+'_predicted_'+model_name], average='binary', pos_label=1)
		res['f1_score_0'] =  f1_score(predictions[variable], predictions[variable+'_predicted_'+model_name], average='binary', pos_label=0)
		res['f1_score_avg'] = f1_score(predictions[variable], predictions[variable+'_predicted_'+model_name], average='weighted')

		res['support_1'] = len(predictions[predictions[variable]==1])
		res['support_0'] = len(predictions[predictions[variable]==0])

		msg = str(res)
		print(msg)
		logger.info(msg)


		results = results.append(pd.DataFrame([res,]))
		results = results[['variable', 'model_name', 'precision_1', 'precision_0','precision_avg',
							'recall_1', 'recall_0', 'recall_avg', 'f1_score_1', 'f1_score_0', 'f1_score_avg',
							'support_1', 'support_0']]









	results.to_pickle('results/results_classifications.pkl')
	results.to_excel('results/results_classifications.xlsx', index=False)

