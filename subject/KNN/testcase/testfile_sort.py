import datetime

starttime = datetime.datetime.now()

import numpy as np
import cv2
#import cv2.cv as cv
import os
import math


#KNN
def kNNClassify(newInput, dataSet, labels, k):
    # step 1
    numSamples = dataSet.shape[0]
    diff = np.tile(newInput, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)
    distance = squaredDist ** 0.5

    # step 2
    sortedDistIndices = np.argsort(distance)
    classCount = {}


    for i in range(k):
        # step 3
        voteLabel = labels[sortedDistIndices[i]]

        # step 4
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # step 5
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex


def testtwoAPI(teststr,trainstr,MR):
    #
    X_test = []
    Y_test = []
    X_train = []
    Y_train = []



    for i in range(0, 10):

        for f in os.listdir((trainstr+'/%s') % i):

            Images = cv2.imread((trainstr+'/%s/%s') % (i, f))

            image = cv2.resize(Images, (256, 256), interpolation=cv2.INTER_NEAREST)

            hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])


            X_train.append(((hist).flatten()))

            Y_train.append(i)


        for f in os.listdir((teststr+'/%s') % i):

            Images = cv2.imread((teststr+'/%s/%s') % (i, f))

            image = cv2.resize(Images, (256, 256), interpolation=cv2.INTER_NEAREST)

            hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])

            X_test.append(((hist).flatten()))

            Y_test.append(i)


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)


    if MR == 22:
        state = np.random.get_state()
        np.random.shuffle(X_train)
        np.random.set_state(state)
        np.random.shuffle(Y_train)

    elif MR == 23:
        num = int(len(X_train[0])/2)
        k = 1/2
        b = 2
        for i in range(0,X_train.shape[0]):
            for j in range(0,num):
                X_train[i][j] = X_train[i][j] * k + b
        for i in range(0,X_test.shape[0]):
            for j in range(0,num):
                X_test[i][j] = X_test[i][j] * k + b

    elif MR == 24:
        state = np.random.get_state()
        for i in range(0,X_train.shape[0]):
            np.random.shuffle(X_train[i])
            np.random.set_state(state)
        for i in range(0,X_test.shape[0]):
            np.random.shuffle(X_test[i])
            np.random.set_state(state)

    elif MR == 25:
        for i in range(0,X_train.shape[0]):
            X_train[i].append(0)
        for i in range(0,X_test.shape[0]):
            X_test[i].append(0)


    predictions = []
    for i in range(X_test.shape[0]):
        predictions_labes = kNNClassify(X_test[i], X_train, Y_train, 5)
        predictions.append(predictions_labes)

    #return predictions
    return 1
