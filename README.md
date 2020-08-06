# Automated Visual Content Analysis using Pre-Trained Models

## About this repository


|||

## Requirements

### Python packages
* pandas
* clarifai
* http
* urllib
* pip install --upgrade google-cloud-vision
* gensim


### Machine Vision APIs

This framework uses Clarifai, Google Cloud Vision and Microsoft. Additional APIs or pre-trained models can be included by extending the models (see helpers/models folder).

You need to have an API key for each API installed in your system. For Clarifai and Microsoft, these keys need to be added to the keys.py file (see below). For Google, you need to download a .json file and add it to your environment path (see details [here](https://cloud.google.com/vision/docs/libraries#client-libraries-install-python)) 


## Scripts

There are three folders with scripts in this repository:
1. **train_models** -> Scripts used to train models based on a subsample manually labelled. 
2. **predict_fullsample** ->  Scripts used to categorize a full sample of images using the best performing models trained in the previous step.
3. **scraper** -> Scripts that illustrate how to collect images from websites for academic research 



### train_models

All files need to be included in a subfolder **source**.

* All images (.png or .jpg) that have been manually coded should be in the folder subsample
	* The image filenames (without the extension) are considered the unique identifiers. They should preferably not have spaces or special characters. The exact same filename should be used in the unique_photo_id in the manual coding.

* The manual coding should be included as an excel file in the subfolder manualcoding. This file should:
    1. Contain a first column named **unique_photo_id** with the unique identifiers (see above)
    2. The following columns are the manually coded variables. These variables must be binary, and be coded as 0 (not present) or 1 (present).
    3. The file should be named **manual_coding.xlsx** 

* The API keys should be added to a file called **keys.py** available. An example can be found at the **keys_template.py** file; 'None' should be substituted by the API key (the API key must be inserted between quotation marks)

* The code (run_routine.py) illustrates how three commercial computer vision API's can be used. The sample code is based on the manual coding of three binary variables (called gen_people, gen_planet and gen_profit). 

It is recommended to review and adapt the code as needed. In some cases, reviewing the actual script being used (in the helpers subfolder) may be helpful. Depending on how complex or large the dataset is, it is easier to comment and uncomment sections of the Python script to run each step at a time.







