import os;
from scipy.misc import imread, imresize
import cv2;
import numpy as np
from random import shuffle


def create_data_sets():
    # Makes path and establishes classifers in dictionary
    imagesPath = "../training_images/"
    datasetPath = "../dataset.txt"
    textFilesPath = "../specialProblem/"
    # trainFilesPath = "../training_imgs/"
    classifiers = {"Chandeliers": 0, "Showerheads":1, "Ceiling Fans":2, "Vanity Lighting": 3,  "Floor Lamps": 4, "Single Handle Bathroom Sink Faucets":5}
    list =[[],[],[],[],[],[]]
    dataset_f = open(datasetPath).read().splitlines()
    for line in dataset_f:
        list[classifiers[line[10:]]].append(line[0:9])


    percentTesting = .20


#
#     # creates traininglist and testinglist and adds approprate amount of images to each list and randomizes them
    trainingList = []
    testingList = []

    for x, array in enumerate(list):
        testingNum = int(percentTesting * len(array))
        for i in range(0, len(array)):
            entry = imagesPath + array[i] + ".jpg" + " " + str(x)
            if(i < testingNum):
                testingList.append(entry)
            else:
                trainingList.append(entry)

    shuffle(trainingList)
    shuffle(testingList)
#     # Creates/overwrites existing text files for training and testing
    training = open(textFilesPath + "train.txt", "w+")
    testing = open(textFilesPath + "test.txt", "w+")
    # writes to training and testing text files
    for entry in trainingList:
        training.write(entry + "\n")
    for entry in testingList:
        testing.write(entry + "\n")

    # Closes the text files
    training.close()
    testing.close()

# create_data_sets()
