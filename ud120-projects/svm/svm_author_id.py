#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

from time import time
#sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
from sklearn.svm import SVC
clf = SVC(C=10000.0, kernel="rbf", gamma=1.0)

#this makes training set smaller and faster
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

clf.fit( features_train, labels_train )
pred = clf.predict( features_test )

#leave this commented, it just throws errors
#prettyPicture(clf, features_test, labels_test)
#plt.show()

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print "accuracy: ", acc

def submitAccuracy():
    return acc