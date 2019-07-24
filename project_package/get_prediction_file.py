import galaxy_image_generator
import model_params
import build_and_train_model
from keras.models import load_model
import keras.losses
import keras.metrics
from keras import backend as K
import numpy as np
import os
from skimage.io import imread
from model_params import params

def rmse(y_true,y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true)))

def get_mean_and_stds(images):
    means = images.mean(axis=(0,1,2),keepdims=True)
    stds = images.std(axis=(0,1,2),keepdims=True)
    return means,stds

def get_norm_vals(params):
    train_files = galaxy_image_generator.get_file_names(params.image_path)
    steps = len(train_files)//params.batch_size
    first = True
    step=0
    for x,y in galaxy_image_generator.batch_generator(params,scale=False):
        step+=1
        if first:
            print(f'Step:1/{steps}',end='\r')
            means,stds = get_mean_and_stds(x)
            means = means/steps
            stds = stds/steps

            first = False
        else:
            batch_means,batch_stds = get_mean_and_stds(x)
            means+=batch_means/steps
            stds+=batch_stds/steps
        print(' '*100,end='\r')
        print(f'Step:{step+1}/{steps}',end = '\r')
        if step == steps:
            return means,stds

def get_test_batches(files,batch_size):
    return [files[i*batch_size:i*batch_size+batch_size] for i in range(len(files)//batch_size+1)]

def get_batch(files,file_path):
    images = []
    for i in files:
        img = imread(f'{file_path}/{i}',as_gray=False)
        images.append(img)
    return np.array(images)

def test_process(images,means,stds,crop=True):
    '''
    preprocessing function used to alter images. Accepts of images
    and returns altered np.array() of images
    '''
    new_images = []

    l = images.shape[1]
    w = images.shape[2]

    if crop:
        for image in images:
            new_image = image[int(0.3*l):int(0.7*l),int(0.3*w):int(0.7*w)]
            new_images.append(new_image)
        images = np.array(new_images)

    images = (images - means)/stds

    return images

def predict_test(model,params,means,stds):

    file_path = params.test_path
    files = sorted(os.listdir(file_path))
    file_batches = get_test_batches(files,100)
    first = True
    i = 1

    for batch in file_batches:
        print(' '*40,end='\r')
        print(f'Progress: {i}/{len(file_batches)}',end='\r')
        images = get_batch(batch,file_path)
        images = test_process(images,means,stds)
        if first:
            predictions = model.predict(images)
            first = False
        else:
            predictions = np.vstack((predictions,model.predict(images)))
        i+=1
    return predictions

def load_galaxy_model():
    keras.losses.rmse = rmse
    keras.metrics.rmse = rmse
    model = load_model('galaxy_morphology_predictor.h5')
    return model

def generate_prediction_file(model_params,calc_norms=False):

    model = load_galaxy_model()
    if calc_norms:
        means,stds = get_norm_vals(model_params)
    else:
        means = np.load('means.npy')
        stds = np.load('stds.npy')
    predictions = predict_test(model,model_params,means,stds)
    save_predictions(predictions,model_params)
    return

def save_predictions(predictions,params):
    sub_df = pd.read_csv(params.sub_path)
    sub_df[sub_df.columns[1:]] = predictions
    sub_df.to_csv('predictions.csv',index=False)
    return sub_df

if __name__ == '__main__':
    model_params = params()
    generate_prediction_file(model_params)
