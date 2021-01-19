#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
########################################################################


from sklearn import tree, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)

feature_tree = tree.DecisionTreeClassifier()
feature_tree = feature_tree.fit(features_train, labels_train)

make_prediction = feature_tree.predict(features_test)

score = accuracy_score(make_prediction, labels_test)
print "\n", "What is the accuracy score? ", score, "\n"

poi_count = sum(labels_test)
print "How many POIs are predicted for the test set? ", poi_count, "\n"

test_set_count = len(labels_test)
print "How many people total are in your test set?", test_set_count, "\n"

hypothetical_accuracy = ((test_set_count-poi_count)/test_set_count)
print "If your identifier predicted 0. (not POI), what would its accuracy be?", hypothetical_accuracy, "\n"


true_positives = 0
for i in range(len(labels_test)):
    if labels_test[i] == 1 and make_prediction[i] == 1:
        true_positives += 1

any_true_positives = ["yes" if true_positives > 0 else "no"]

print "Do you get any true positives? ", any_true_positives, "\n"

poi_precision = precision_score(make_prediction, labels_test)
print "poi precision: ", poi_precision, "\n"

poi_recall = recall_score(make_prediction, labels_test)
print "poi recall: ", poi_recall, "\n"


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

true_predictions = 0.
for i in range(len(true_labels)):
    if true_labels[i] == 1 and predictions[i] == 1:
        true_predictions += 1

print "How many true positives are there?", true_predictions, "\n"

true_negatives = 0.
for i in range(len(true_labels)):
    if true_labels[i] == 0 and predictions[i] == 0:
        true_negatives += 1

print "How many true negatives are there?", true_negatives, "\n"

false_positives = 0.
for i in range(len(true_labels)):
    if true_labels[i] == 0 and predictions[i] == 1:
        false_positives += 1

print "How many false positives are there?", false_positives, "\n"

false_negatives = 0.
for i in range(len(true_labels)):
    if true_labels[i] == 1 and predictions[i] == 0:
        false_negatives += 1

print "How many false negatives are there?", false_negatives, "\n"

hypothetical_precision = float(true_predictions/(true_predictions + false_positives))
print "What's the precision of this classifier?", hypothetical_precision, "\n"

hypothetical_recall = float(true_predictions/(true_predictions + false_negatives))
print "What's the recall of this classifier?", hypothetical_recall, "\n"