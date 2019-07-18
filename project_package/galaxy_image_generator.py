from skimage.io import imread
import pandas as pd
import numpy as np
from skimage.io import imread
import gc
import os
from model_params import params

def label_loader(file):
    '''Loads survey responses for morphology classification task'''
    labels = pd.read_csv(file)
    labels.set_index('GalaxyID',inplace = True)
    print('Labels Successfully Loaded')
    return labels

def preprocess(images,crop=True,scale=True):
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

    if scale:
        means = images.mean(axis=(0,1,2),keepdims=True)
        stds = images.std(axis=(0,1,2),keepdims=True)
        images = (images - means)/stds

    return images

def get_file_names(folder):
    '''returns file names in a given folder within the current working
        directory'''
    return os.listdir(folder)

def get_images_and_labels(images_path,index,labels):
    '''takes in file path to images,image numbers (index), and labels. Then
        returns images as np.array() and labels as pandas DataFrame'''
    images = []
    for i in index:
        img = imread(f'{images_path}/{i}.jpg',as_gray=False)
        images.append(img)
    labels = labels.loc[index]
    return np.array(images),labels

def get_index(images_path,n_images):
    '''removes .jpg ending and returns file labels as integers:
    eg '100008.jpg' -> int(100008)
    '''
    return [int(i[:-4]) for i in get_file_names(images_path)[:n_images]]

def make_batches(index,batch_size):
    '''accepts index for training images and returns batches for one epoch'''
    index = np.random.choice(index,size=len(index))
    steps_per_epoch = len(index)//batch_size
    batches = [index[i*batch_size:i*batch_size+batch_size] for i in range(steps_per_epoch)]
    return batches

def batch_generator(params):
    '''accepts model parameters from model_params.py file and returns a
        generator which can be passed to keras fit_generator function'''
    image_path = params.image_path
    label_path = params.label_path
    n_images = params.n_images
    batch_size = params.batch_size

    labels = label_loader(label_path)
    index = get_index(image_path,n_images)
    while True:
        batches = make_batches(index,batch_size)
        for batch_index in batches:
            batch_x,batch_y = get_images_and_labels(image_path,batch_index,labels)
            batch_x = preprocess(batch_x)
            yield( batch_x,batch_y)

if __name__ == '__main__':
    print('Testing Process')
    print('-'*40)
    i = 0
    for x,y in batch_generator(params()):
        print(f'Batch {i+1} Test')
        print('Image Shape:',x.shape)
        print('Label Shape:',y.shape)
        print('...')
        i+=1
        if i ==2:
            break
    print('-'*40)
    print('All Tests Complete')
    print('Generating Batchs Normally')
