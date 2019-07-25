# Image Classification for Galaxy Morphologies

---

This project involoved the development of a custom convolutional neural network in order to classify different galaxy morphologies.

The final model can be fully trained using the files contained within the **project_package** folder and the publically available data set (see below).

## Data Set:
---

Data used in this project is publically available on kaggle at:

https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview

The data set contained images as well as normalized survey responses for the Galaxy Zoo Survey:

![](images/GalaxtZooTree.png)

### Images

Initial Images in the data set were all formatted to 412x412x3:

![](images/raw_images_sample.png)

### Labels

Survey Responses were given in csv format:

![](images/sample_labels.png)

## Prerequisites

This repo is executed solely in python and requires the following imports to run:

![](images/import.png)

## Project Summary:

As the number of images being taken to survey surrounding galaxies increases the need for classifying particular galaxies increases as well. Currently many galaxies are classified by crowd sourcing volunteers and having them identify certain shapes and morphologies by hand. Though this approach has worked in the past it becomes less feasible as the number of images moves into the millions and millions. This project looks at solving this problem through the use of Convolutional Neural Networks in order classify these images in a way that is more scalable to massive data sets.

## The Model:

The architecture used ended up being based on the architecture of AlexNet, one of the first CNNs to demonstrate their usefulness for image classification tasks. The final architecture and parameters are shown below.

![](project_package/images/final_model_summary.png)

The model utilizes sub sampling on the first three convolutional layers in order to make training feasible on a conventional laptop. The model converges after ~7 Passes through the data settling on a RMSE around 10.7.

![](project_package/images/loss_history.png)


## Results:

The final model was trained on 55000 images and valiated on the remaining 6000. The submission scores 0.10758 on kaggle which would put it in 68th position (Bronze). 

![](project_package/images/kaggle_score.png)


![](images/model_performance.png)

## Author
- Liam Wood Roberts
