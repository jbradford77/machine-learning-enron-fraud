#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#############################################################
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import linear_model

from outlier_cleaner import outlierCleaner
from sklearn import model_selection
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from class_vis import prettyPicture

import copy
import pylab as pl

from time import time
from email_preprocess import preprocess

############################################################

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#features_list = ['poi','salary'] # You will need to use more features

features_list = ['poi','salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'director_fees' ] 

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person',
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

POI_label = ['poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#total number of data points
num_data_points = len(data_dict)
print "\n - total number of data points: ", num_data_points

#allocation across classes (POI/non-POI)
poi_count = 0
for key in data_dict:
    if data_dict[key]['poi']==1:
        poi_count +=1
non_poi_count = 0
for key in data_dict:
    if data_dict[key]['poi']==0:
        non_poi_count +=1
print "\n - allocation across classes (POI/non-POI): ", poi_count,"/",non_poi_count

#number of features used
feature_count = 0
for key in range(len(data_dict[key])):
    feature_count += 1
print "\n - number of features used: ", feature_count

print "\n are there features with many missing values? etc. Yes, and here they are"
no_salary = sum(1 for d in data_dict.values() if d['salary'] == 'NaN')
print "\n - Number people no salary: ", no_salary

no_payments = sum(1 for d in data_dict.values() if d['total_payments'] == 'NaN')
print "\n - Number NaN payments: ", no_payments

percent_no_payments = float(no_payments)/len(data_dict)
print "\n - percent NaN payments: ", percent_no_payments

poi_nan_payment_count = 0
poi_nan_payment = 0
for key in data_dict:
    if data_dict[key]['poi']==1:
        poi_nan_payment_count += 1
        if data_dict[key]['total_payments']=='NaN':
            poi_nan_payment += 1

print "\n - Number of POIs no payments: ", poi_nan_payment

percent_poi_nan_payment = float(poi_nan_payment)/poi_nan_payment_count
print "\n - percent poi NaN payments: ", percent_poi_nan_payment


#stuff nobody exactly asked for but here it is anyway
feature_sample = 0
for key in data_dict:
    print "\n - Sample person from dataset: ", key
    #print "\n - Number of features per person: ", len(data_dict[key])
    print "\n - Features: ", data_dict[key]

    feature_sample +=1
    if feature_sample == 1:
        break

df=pd.read_csv('poi_names.txt', header=None, skiprows=[0], sep=')')
df.columns = ['POI', "Name"]
df = df.replace(['\('], [''], regex=True)
#num_poi=df.count()
num_poi = df.POI.value_counts()
num_poi_we_have_emails = df.POI.value_counts()["y"]

print "\n - Total number of POIs: \n", num_poi
print "\n - Total number of POIs where we have their emails: ", num_poi_we_have_emails
print "\n",df.sort_values('POI', ascending=False)



### Task 2: Remove outliers


data_dict.pop('TOTAL')

#for key in data_dict:
#    print key
data = featureFormat(data_dict, features_list)
'''
for point in data:
    salary = point[1]
    bonus = point[5]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


for point in data:
    all_stonks = point[8]
    stonks_options = point[10]
    plt.scatter( all_stonks, stonks_options )


plt.xlabel("total stock value")
plt.ylabel("stock options exercised")
plt.show()
'''


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#trying to find a relation between percent of emails sent
#by each person to a poi and their stock values. Maybe something here?
#I'm curious if people were asking about when to cash in options
#emails = data_dict commented this and updated code when I noticed the my_dataset

for point in my_dataset:
    to_poi = my_dataset[point]['from_this_person_to_poi']
    all_msgs_from_person = my_dataset[point]['from_messages']
    stonks = my_dataset[point]['total_stock_value']
    if to_poi != "NaN" and to_poi != 0 and all_msgs_from_person != "NaN" and all_msgs_from_person != 0:
        my_dataset[point]['percent_to_poi'] = float(to_poi)/float(all_msgs_from_person)
        percent_emails_to_poi = my_dataset[point]['percent_to_poi']
        #print "percent emails POI: ", point, percent_emails_to_poi
    if stonks != 0 and stonks != "NaN":
        people_with_stonks = stonks
        #print "people with stonks: ", point, people_with_stonks
    plt.scatter( percent_emails_to_poi, people_with_stonks )

plt.xlabel("percent emails to POIs")
plt.ylabel("people with stocks")
plt.legend(my_dataset.keys(), fontsize='xx-small', )
#plt.figure(figsize=(20,10)) 
#plt.show()

#more emails from poi means they liked a person and gave bigger bonus?
#filtered_data = {}
for point in my_dataset:
    from_poi = my_dataset[point]['from_poi_to_this_person']
    all_msgs_to_person = my_dataset[point]['to_messages']
    bonus = my_dataset[point]['bonus']
    if from_poi != "NaN" and from_poi != 0 and all_msgs_to_person != "NaN" and all_msgs_to_person != 0:
        my_dataset[point]['percent_from_poi'] = float(from_poi)/float(all_msgs_to_person)
        percent_emails_from_poi = my_dataset[point]['percent_from_poi']
    else:
        my_dataset[point]['percent_from_poi'] = 0
    if percent_emails_from_poi > 0.10:
        high_percent_from_poi = percent_emails_from_poi
        print "10% or more emails from POI: ", point, high_percent_from_poi
    if bonus != 0 and bonus != "NaN":
        people_with_bonus = bonus
        #print "people with bonus: ", point, people_with_bonus
    my_dataset[point]
    plt.scatter( percent_emails_from_poi, people_with_bonus )
    


plt.xlabel("percent emails from POIs")
plt.ylabel("people with bonus")
plt.legend(my_dataset.keys(), fontsize='xx-small', )
#plt.figure(figsize=(20,10)) 
#plt.show()



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn import tree, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

make_prediction = clf.predict(features_test)

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

############################################################

###############################################################

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="x")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    #plt.legend(my_dataset.keys(), fontsize='xx-small', )
    plt.show()


exercised_options = []
for i in my_dataset:
    record = my_dataset[i]["total_stock_value"]
    if record != "NaN":
        exercised_options.append(record)

print "\n minimum stock options exercised: ", min(exercised_options)
print "\n maximum stock options exercised: ", max(exercised_options)

bonus_not_zero = []
for b in my_dataset:
    record = my_dataset[b]["bonus"]
    if record != "NaN":
        bonus_not_zero.append(record)

print "\n minimum bonus: ", min(bonus_not_zero)
print "\n max bonus: ", max(bonus_not_zero)

for point in my_dataset:
    to_poi = my_dataset[point]['from_this_person_to_poi']
    all_msgs_from_person = my_dataset[point]['from_messages']
    if to_poi != "NaN" and to_poi != 0 and all_msgs_from_person != "NaN" and all_msgs_from_person != 0:
        my_dataset[point]['percent_to_poi'] = float(to_poi)/float(all_msgs_from_person)
        #percent_emails_to_poi = my_dataset[point]['percent_to_poi']
    else:
        my_dataset[point]['percent_to_poi'] = 0


# min-max of a percentage probably wasn't the best idea. Don't uncomment this trash

email_to_poi_not_zero = []
for b in my_dataset:
    record = my_dataset[b]["percent_to_poi"]
    if record != "NaN" and record != 0:
        email_to_poi_not_zero.append(record)

print "\n minimum email to poi: ", min(email_to_poi_not_zero)
print "\n max email to poi: ", max(email_to_poi_not_zero)


features_list = ['poi','salary', 'total_payments','loan_advances', 'bonus', 
'restricted_stock_deferred', 'deferred_income','total_stock_value', 
'exercised_stock_options', 'restricted_stock', "percent_to_poi", 'percent_from_poi'] 


### the input features in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "bonus"
feature_2 = "total_stock_value"
#feature_3 = "salary"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(my_dataset, features_list )
poi, finance_features = targetFeatureSplit( data )


### if you add a feature you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
#plt.legend(my_dataset.keys(), fontsize='xx-small', )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
rescaled_features = scaler.fit_transform(finance_features)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(rescaled_features)
pred = kmeans.predict(rescaled_features)

try:
    Draw(pred, rescaled_features, poi, mark_poi=True, name="scaled_clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"

bonus_and_options = np.array([[70000., 8000000.]])
rescaled_bonus_options = scaler.transform(bonus_and_options)

print "Rescaled $8,000,000 bonus and $1,000,000 exercised stock options: ", rescaled_bonus_options

#####################################################################
'''
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = GaussianNB()

clf = clf.fit(features_train, labels_train)
GaussianNB(labels_test)

'''
###############################################################

from sklearn.svm import SVC
clf = SVC(C=1000.0, kernel="rbf", gamma=200.0)

#this makes training set smaller and faster
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf.fit( features_train, labels_train )
pred = clf.predict( features_test )

#leave this commented, it just throws errors
#prettyPicture(clf, features_test, labels_test)
plt.show()

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print "\n accuracy: ", acc

true_positives = 0
for i in range(len(labels_test)):
    if labels_test[i] == 1 and make_prediction[i] == 1:
        true_positives += 1

any_true_positives = ["yes" if true_positives > 0 else "no"]
print "Do you get any true positives? (new) ", any_true_positives, "\n"

poi_precision = precision_score(make_prediction, labels_test)
print "poi precision: (new)", poi_precision, "\n"

poi_recall = recall_score(make_prediction, labels_test)
print "poi recall: (last print)", poi_recall, "\n"


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: or a lovely 404,
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
from sklearn import svm, datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#this makes training set smaller and faster
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

'''
clf = svm.SVC(kernel='linear', C=1).fit(features_train, labels_train)
sss_score = clf.score(features_test, labels_test)
print sss_score
'''



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
