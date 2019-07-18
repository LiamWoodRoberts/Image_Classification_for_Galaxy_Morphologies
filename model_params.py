class params:
    '''
    Class Containing all variables needed to execute project
    '''
    def __init__(self):
        '''
        variables:

        label_path - absolute path to labels (with label file name)
        image_path - absolute path to images (with image folder name)
        n_images - the number of images to train on
        batch_size - what batch size to use when training CNN
        epochs - How many passes through the data set when training model

        '''
        # Path to Folder
        self.label_path = '.../training_solutions_rev1.csv'
        self.image_path = '.../images_training_rev1'

        # Number of Images to Train On
        self.n_images = 1000

        # Params for Training
        self.batch_size = 100
        self.epochs = 5
