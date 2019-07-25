import galaxy_image_generator
from keras.models import Sequential, Model, Input
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Conv1D,MaxPooling2D,Dropout,SpatialDropout2D,BatchNormalization
from model_params import params
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers import ReLU
import pandas as pd
from keras.callbacks import ModelCheckpoint

def create_model(input_shape,output_shape):
    '''Creates a CNN based on the ALEXNET architecture'''
    model = Sequential()
    model.add(Conv2D(32,(11,11),padding='same',
                                activation='relu',
                                strides=(4,4),
                                input_shape=input_shape))

    model.add(MaxPooling2D((3, 3),strides = (2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (5, 5),activation='relu'))
    model.add(MaxPooling2D((3, 3),strides = (2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3),activation='relu'))
    model.add(Conv2D(128, (3, 3),activation='relu'))
    model.add(Conv2D(128, (3, 3),activation='relu'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape))
    model.add(ReLU(max_value=1))
    return model

def rmse(y_true,y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true)))

def train_model(model,train_generator,valid_generator,params,save=True):
    '''Accepts a keras.model, a generator object and model parameters (from
        model_params.py).

        Returns a trained keras.model'''
    model.compile(optimizer='Adam',loss=rmse,metrics=[rmse])

    # From Model Params
    n_training_images = params.n_training_images
    n_valid_images = params.n_valid_images

    batch_size = params.batch_size
    epochs = params.epochs

    train_steps_per_epoch= n_training_images//batch_size
    valid_steps_per_epoch = n_valid_images//batch_size

    early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.0001,
                           patience=3,
                           mode='min',
                           verbose=1)

    checkpointer = ModelCheckpoint('galaxy_morphology_predictor.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True)

    model_history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        validation_data = valid_generator,
                        validation_steps = valid_steps_per_epoch,
                        callbacks=[early_stop,checkpointer])

    return model_history

def get_batch_shape(generator):
    '''returns shape of images and labels outputed from generator'''
    for x,y in generator:
        return x[0].shape,y.shape[1]

def save_training_info(model):
    history = pd.DataFrame()
    history['val_loss'] = model.history['val_loss']
    history['trn_loss'] = model.history['loss']
    history['epochs'] = [i for i in range(1, len(history) + 1)]
    history.to_csv('model_history.csv',index=False)
    return

def run_project(params,save=True):
    '''Carries out the following process to build an image classifier for
        galaxy morphologies.

        1.Creates batch generator object with params in model_params.py
        2. Creates keras Model object
        3. Trains keras Model object using the batch generator object and
        params in model_params.py
        4. Returns trained keras Model image classifier

        '''

    # Get Custom Generator
    train_gen = galaxy_image_generator.batch_generator(params)
    valid_gen = galaxy_image_generator.batch_generator(params,valid=True)

    # Shape from Generator
    input_shape,output_shape = get_batch_shape(train_gen)

    # Create CNN
    model = create_model(input_shape,output_shape)

    print('-'*40)
    print('Current Model Architecture')
    print(model.summary())

    # Train Model
    trained_model = train_model(model,train_gen,valid_gen,params,save=save)
    save_training_info(trained_model)
    return trained_model

if __name__ == '__main__':

    print('Testing Model Training Process...')
    trained_model = run_project(params(),save=True)
    print('~~~ Model Successfully Trained ~~~')

