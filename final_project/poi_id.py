#!/usr/bin/python
from __future__ import division
import sys
import pickle
import numpy as np
sys.path.append("../tools/")
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from feature_format import featureFormat, targetFeatureSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as mp
from tester import dump_classifier_and_data


PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
###################
def FeaturesList(data): # add features from existing features in financial data set using poi "Kenneth Lay" as an example
    features_list = ['poi','salary']
    for key, value in data["LAY KENNETH L"].iteritems():
        if (key not in features_list and key != 'email_address'):
            features_list.append(key)
    return features_list
###################
### Store to my_dataset for easy export below.

my_dataset = data_dict

#####################
def AddFeatures(features_list, my_dataset):
    """Add new features to existing dataset
    args:
        features_list: list containing existing features
        my_dataset: dictionary containing Enron data
    """
    total_messages = ['from_messages', 'to_messages']
    total_messages_with_poi = ['from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']
    stock = ['total_stock_value', 'exercised_stock_options']
    loan = ['total_payments', 'loan_advances']
    
    for person, value in my_dataset.iteritems():  
	      
        my_dataset[person]['total_messages'] = 0
        my_dataset[person]['total_messages_with_poi'] = 0
	my_dataset[person]['message_shared_fraction'] = 0
	my_dataset[person]['excerised_stock_ratio'] = 0


        my_dataset[person]['total_messages']=np.nansum(np.array([my_dataset[person][key] for key in total_messages]).astype(np.float))
	my_dataset[person]['total_messages_with_poi']=np.nansum(np.array([my_dataset[person][key] for key in total_messages_with_poi]).astype(np.float))
	  
        if my_dataset[person]['total_messages_with_poi'] != 'NaN' and [0, 'NaN'].count(my_dataset[person]['total_messages'])==0:
            my_dataset[person]['message_shared_fraction'] = float(my_dataset[person]['total_messages_with_poi'])/ my_dataset[person]['total_messages']

        if my_dataset[person]['exercised_stock_options'] != 'NaN' and [0, 'NaN'].count(my_dataset[person]['total_stock_value'])==0:
            my_dataset[person]['excerised_stock_ratio'] = float(my_dataset[person]['exercised_stock_options']) / my_dataset[person]['total_stock_value'] 

    return FeaturesList(my_dataset), my_dataset

#######################
def FormatFeatures(dataset, features_list):
	data = featureFormat(dataset, features_list, sort_keys=True)
	labels, features = targetFeatureSplit(data)
	return labels, features
#######################
def ScaleFeatures(features):
	scaler = MinMaxScaler()
	features = scaler.fit_transform(features)
	return features
#######################
def TrainTestSplit(labels, features):

    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])
    return features_train, features_test, labels_train, labels_test
#######################
def test_classifier(clf, labels, features):

    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        # fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)

        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*float(true_positives + true_negatives)/total_predictions
        precision = 1.0*float(true_positives)/(true_positives+false_positives)
        recall = 1.0*float(true_positives)/(true_positives+false_negatives)
        f1 = 2.0 * float(true_positives)/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) *float(precision*recall)/(4*precision + recall)
 	
	'''
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
	'''
        return precision, recall, f1		
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

#######################

def tuneAlgorithm(clf, features, labels):
    """Prints the best params for the RandomForestClassifier based on the
    results of the GridSearchCV
    """
    score_metric = 'precision'
    
    if clf == RandomForestClassifier():

        params = {"n_estimators": [10], "min_samples_split": range(1, 11), "min_samples_leaf": range(1, 11),"criterion": ["gini", "entropy"]}
    else:
        params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8 , 50], 'weights': ['uniform', 'distance'], 'p': [1, 1.5, 2], 'leaf_size': [1, 2, 3, 10, 50],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    
    search = GridSearchCV(estimator=clf, param_grid=params, scoring=score_metric, n_jobs=1, refit=True, cv=10)
    search.fit(features, labels)
    print search.best_params_
    print search.best_score_

#########################
def TopFeatures(clf, features_list, labels, features, n_features):
    """ Given a list of all features, returns the best n_features number
	of features as a list
    """

    features_train, features_test, labels_train, labels_test = TrainTestSplit(labels, features)    
    if clf == RandomForestClassifier():
        best = clf.fit(features_train, labels_train)
        importance = best.feature_importances_
    else:
        best = SelectKBest(k=n_features) 
        best.fit_transform(features_train, labels_train)
        importance = best.scores_

    features_with_scores = sorted(zip(features_list[1:], importance), key=lambda l: l[1], reverse=True)
    best_features_list=[x[0] for x in features_with_scores[:n_features]]
    best_features_list.insert(0,'poi')

    return best_features_list    

def FeaturesNumberScores(clf, features_list, my_dataset):
    precision_scores=[]
    recall_scores=[]
    f1_scores=[]
    number_of_features=[]
    labels_full, features_full = FormatFeatures(my_dataset, features_list)
    for i in range(1, len(features_list)):
        best_features_list = TopFeatures(clf, features_list, labels_full, features_full, i)
        labels, features = FormatFeatures(my_dataset, best_features_list)
        #features = ScaleFeatures(features)
        precision, recall, f1=test_classifier(clf, labels, features)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
	number_of_features.append(i)
        #print i, precision, recall, f1
    mp.plot(number_of_features, precision_scores, marker='*', label="Precision")
    mp.plot(number_of_features, recall_scores, marker='*', label="recall")
    mp.plot(number_of_features, f1_scores, marker='*', label="F1")
    mp.xlabel("Number of Features")
    mp.ylabel("Performance Metric Scores")
    mp.legend()
    mp.show()
############################################################################

### Task 1: Select what features you'll use.
features_list=FeaturesList(my_dataset)

### Task 2: Remove outliers
my_dataset.pop('TOTAL', 0)
my_dataset.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### Task 3: Create new feature(s)
updated_features_list, my_dataset = AddFeatures(features_list, my_dataset)

features_list = updated_features_list

### Task 4: Try a varity of classifiers

### Extract features and labels from dataset for local testing
labels, features = FormatFeatures(my_dataset, features_list)
#features = ScaleFeatures(features)
#clf = RandomForestClassifier()
clf = KNeighborsClassifier()
n_features=12
best_features_list = TopFeatures(clf, features_list, labels, features, n_features)
#print best_features_list

### Task 5: Tune the classifier to achieve better than .3 precision and recall 

#tuneAlgorithm(clf, features, labels)

clfRF = RandomForestClassifier(n_estimators = 10, max_features='auto', min_samples_split=1, criterion='gini', min_samples_leaf=1)
clfKNN = KNeighborsClassifier(n_neighbors=3, weights='uniform', leaf_size=1, algorithm='auto', p=1)
clf=clfKNN
#FeaturesNumberScores(clf, features_list, my_dataset)

labels, features = FormatFeatures(my_dataset, best_features_list)
test_classifier(clf, labels, features)


### Task 6: Dump the classifier, dataset, and features_list so anyone can
### check the results.
features_list=best_features_list 
dump_classifier_and_data(clf, my_dataset, features_list)


