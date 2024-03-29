# Kaggle - Galaxy Morphologies
---
Submission for the Galaxy Zoo Challenge on Kaggle (http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge).

### File Summaries:

**galaxy_image_generator.py:** File used to create batches for training keras model. Is called by the build_and_train_model.py file.

**build_and_train_model.py:** File used to train new networks using the described architecture on batches of images. Is suitible for running on a conventional laptop. Outputs a saved model to *galaxy_morphology_predictor.h5* file.

**get_prediction_file.py:** Calls the *galaxy_morphology_predictor.h5* file to generate predictions on the test set (uses batch prediction to save memory).

**model_params.py:** Holds parameters for model training as well as folder location. When running locally "folder_path" must be changed to the absolute path location to data files.

**galaxy_morphology_predictor.h5:** Saved keras model for kaggle competiton. Generated by *build_and_train_model.py* and called by *get_prediction_file.py*.

**means.npy, stds.npy:** Training set mean and std values used to scale image data for train and test set.

**predictions.csv:** Prediction file for kaggle competition. Default places 68th / Bronze postion.

**requirements.txt**: Text file indicating required python environment packages to run all files included in project_package folder. 


### Requirements
Project is executed entirely in python. Requires, keras, sklearn, numpy,pandas, and skimage. More information available in requirements.txt. Environment can be downloaded through:

<code> pip install -r requirements.txt </code>

### Download Data:

All data used is available publically at:

https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data

### Update model_params.py:

The only value that needs to be updated is the folder_path. Set this to whatever the absolute path is for the downloaded data.

### Model Training:

Models can be trained by running the file **build_and_train_model.py**. Parameters for training can be altered by adjusting their values in the **model_params.py** file. Architecture for model is available below:

![](images/final_model_summary.png)


### Generating Predictions:

Predictions can be generated by running **get_prediction_file.py**. The default model is **galaxy_morphology_predictor.h5** but will be replaced by any model trained using the **build_and_train_model.py** file. Predictions take ~20min to generate on a laptop and place 68th (Bronze).

![](images/kaggle_score.png)

