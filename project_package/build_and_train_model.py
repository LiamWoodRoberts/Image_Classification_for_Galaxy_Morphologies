import galaxy_image_generator
from keras.models import Sequential, Model, Input
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Conv1D,MaxPooling2D,Dropout,SpatialDropout2D,BatchNormalization
from model_params import params

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
    model.add(Dense(output_shape, activation='softmax'))
    return model

def train_model(model,generator,params,save=True):
    '''Accepts a keras.model, a generator object and model parameters (from
        model_params.py).

        Returns a trained keras.model'''
    model.compile(optimizer='Adam',loss='mean_squared_error',metrics=['mse'])

    # From Model Params
    n_images = params.n_images
    batch_size = params.batch_size
    epochs = params.epochs

    steps_per_epoch=n_images//batch_size

    model.fit_generator(generator=generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs)
    if save:
        model.save('galaxy_morphology_predictor.h5')
        
    return model

def get_batch_shape(generator):
    '''returns shape of images and labels outputed from generator'''
    for x,y in generator:
        return x[0].shape,y.shape[1]

def run_project(params):
    '''Carries out the following process to build an image classifier for
        galaxy morphologies.

        1.Creates batch generator object with params in model_params.py
        2. Creates keras Model object
        3. Trains keras Model object using the batch generator object and
        params in model_params.py
        4. Returns trained keras Model image classifier

        '''

    # Get Custom Generator
    custom_gen = galaxy_image_generator.batch_generator(params)

    # Shape from Generator
    input_shape,output_shape = get_batch_shape(custom_gen)

    # Create CNN
    model = create_model(input_shape,output_shape)

    print('-'*40)
    print('Current Model Architecture')
    print(model.summary())

    # Train Model
    trained_model = train_model(model,custom_gen,params)

    return trained_model

if __name__ == '__main__':

    print('Testing Model Training Process...')
    trained_model = run_project(params())
    print('~~~ Model Successfully Trained ~~~')
