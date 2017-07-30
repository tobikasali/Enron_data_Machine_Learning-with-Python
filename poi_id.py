#!/usr/bin/python

import sys
import pickle
import math
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from operator import itemgetter, attrgetter
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz
#from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi', "restricted_stock", "expenses",  "fraction_from_poi","shared_receipt_with_poi"] #  # You will need to use more features
features_list = ['poi',   "fraction_from_poi","shared_receipt_with_poi"]
##
"""
 feature list used for Testing classifiers
 
 features_list = ["poi", "restricted_stock", "director_fees","exercised_stock_options","total_stock_value","bonus", "salary", "total_payments","expenses", "long_term_incentive", "fraction_from_poi","fraction_to_poi","shared_receipt_with_poi"]
 
"""

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop("TOTAL", 0)
data_dict.pop("BELFER ROBERT", 0)

for key, value in data_dict.items():
    if key == "BHATNAGAR SANJAY" :
       value['restricted_stock'] = 'NaN'

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
### Add new features to data set
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity
    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0
    test_f = math.isnan(float(poi_messages)) # Test if the values are Nan
    test_m = math.isnan(float(all_messages))  # Test if the values are Nan
    
    if (all_messages == 0 or poi_messages ==0):        
        fraction = 0
    elif not (test_f or test_m):
        fraction = float((poi_messages))/ float((all_messages))    
    else:
        fraction = 0        
    return fraction

l_fraction_to_poi = []
l_fraction_from_poi = []
for key,value in data_dict.items():
    value ['fraction_from_poi'] =  computeFraction(value['from_this_person_to_poi'], value['from_messages'])
    value ['fraction_to_poi'] =    computeFraction(value['from_poi_to_this_person'], value['to_messages'])

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

feature_names = features_list[1:len(features_list)]
label_names = ["NON_POI", "POI"]

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB


## Algorithm 1 -GuassinNB - Algorithm with SelectKbest and Parameter Tuning
""" 


select = SelectKBest()
scaling = MinMaxScaler()
clf = GaussianNB()
feature_names = features_list[1:len(features_list)]
label_names = ["NON_POI", "POI"]

steps = [('feature_selection', select), ("scaler",scaling),('naive_bayes', clf)]
pipeline = sklearn.pipeline.Pipeline(steps)

parameters = dict(feature_selection__k=[3,4,5,6,7,8,9,10], 
              naive_bayes__priors = [None])


"""
### Algorithm 2 Support Vector Classifier  with SelectKbest - Algorithm and Parameter Tuning

"""

select = SelectKBest()
clf = SVC()
feature_names = features_list[1:len(features_list)]
label_names = ["NON_POI", "POI"]

steps = [("feature_selection", select),("my_classifier", clf)]

pipeline = sklearn.pipeline.Pipeline(steps)
parameters = dict(feature_selection__k=[3,4,5,6,7,8,9,10], 
              my_classifier__C= [0.1, 1, 2, 4, 6, 8, 10],
              my_classifier__kernel=["rbf"],
              my_classifier__gamma = [0.01, 0.1, 1, 10.0, 50.0, 100.0],
              my_classifier__class_weight =["balanced"])




### Tune your classifer 

sss = StratifiedShuffleSplit(n_splits=10, test_size = 0.3,random_state = 42)
cv = GridSearchCV(pipeline, param_grid = parameters, cv=sss, scoring='f1')
cv.fit(features_train,labels_train )


clf = cv.best_estimator_

best_features = clf.named_steps['feature_selection'].get_support()

for position, value in enumerate(best_features):
    if value == 1:
        print feature_names[position]
print "Predicting the people names on the testing set"

"""

### Algorithm 3 Decsion Tree with SelectKbest - Algorithm and Parameter Tuning
"""

select = SelectKBest()
scaling = MinMaxScaler()
clf = DecisionTreeClassifier()



label_names = ["NON_POI", "POI"]

steps = [("feature_selection", select), ("scaler",scaling),("my_classifier", clf)]

pipeline = sklearn.pipeline.Pipeline(steps)
parameters = dict(feature_selection__k=[3,4,5,6], 
              my_classifier__max_depth= [None, 1 ,2, 5, 10],
              my_classifier__min_samples_split=[2, 3, 4, 5, 10],
              my_classifier__criterion = ["gini", "entropy"],
              my_classifier__class_weight =["balanced"])


"""
#### Algorithm 4 Decision Tree wiht feature Importances - Algoruthm and Parameter Tuning

"""
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 5,10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }

"""

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
     
    ###############################################################################

### Parameter tuning for tested Classifiers - GuassianNB, SVC and Dection Tree using SelecrKBest

"""

sss = StratifiedShuffleSplit(n_splits=10, test_size = 0.3,random_state = 42)
cv = GridSearchCV(pipeline, param_grid = parameters, cv=sss, scoring='f1')
cv.fit(features_train,labels_train )

## Select best estimator

clf = cv.best_estimator_

### get Best features

best_features = clf.named_steps['feature_selection'].get_support()
#features.columns[features.transform(np.arange(len(features.columns)))]

for position, value in enumerate(best_features):
    if value == 1:
        print feature_names[position]

"""


###  parametr tuning using Decision tree and feature_importances


"""
sss = StratifiedShuffleSplit(n_splits=10, test_size =0.3,random_state = 42)
gs = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), param_grid, cv=sss)
#gs = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=sss)

gs.fit(features_train,labels_train)


print "Best estimator found by grid search:"
print gs.best_estimator_

###############################################################################

# Quantitative evaluation of the model quality on the test set
#pred = gs.predict(features_test)
#print "done in %0.3fs" % (time() - t0)


## get the best classfier
clf = gs.best_estimator_

### Get the best important features

important_features = clf.feature_importances_

for position,score in enumerate(important_features):
        
    print feature_names[position]
    print position
    print score

"""

#print classification_report(labels_test, pred, target_names=label_names)

#print confusion_matrix(labels_test, pred)

#### Final Algorithm & classifer using Decision Tree and chosen parameters

#clf = DecisionTreeClassifier(class_weight='balanced', criterion='gini',
#            max_depth=5,  max_leaf_nodes=20,
#            min_samples_leaf=1,
#            min_samples_split=2)
#
#clf.fit(features_train,labels_train )
#
#important_features = clf.feature_importances_
#
#for position,score in enumerate(important_features):
#        
#    print feature_names[position]
#    print position
#    print score



clf = DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
            max_depth=2,  max_leaf_nodes=None,
            min_samples_leaf=1,
            min_samples_split=2)

clf.fit(features_train,labels_train )

important_features = clf.feature_importances_

for position,score in enumerate(important_features):
        
    print feature_names[position]
    print position
    print score

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)