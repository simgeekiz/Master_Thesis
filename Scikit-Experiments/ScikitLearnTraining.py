import numpy as np

from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score as scikit_f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import pickle

import pprint
import os
import sys
sys.path.append('../')
sys.path.append('../../')
from load_data import load_data

## Load Data
train_data, valid_data, test_data, metadata = load_data('/kuacc/users/simgebasar/workspace/Master_Thesis')

train_data = train_data[:1]
valid_data = valid_data[:1]
test_data = test_data[:1]

main_folder = '/kuacc/users/simgebasar/workspace/Master_Thesis/Scikit-Experiments'

number_stopwords = [str(i) for i in range(10001)] + ['0'+str(i) for i in range(100)] + ['000']

scoring = 'f1_macro'
n_jobs=10

tra_sents = np.array([sentence['sentence']
     for article in train_data
     for sentence in article['sentences']])
y_tra = np.array([sentence['label'] for article in train_data for sentence in article['sentences']])

opt_sents = np.array([sentence['sentence']
     for article in (train_data + valid_data)
     for sentence in article['sentences']])

y_opt = np.array([sentence['label'] for article in (train_data + valid_data) for sentence in article['sentences']])

test_sents =  np.array([sentence['sentence']
     for article in test_data
     for sentence in article['sentences']])

y_test = np.array([sentence['label'] for article in test_data for sentence in article['sentences']])


vectorizer = TfidfVectorizer(min_df=0.001, max_df=0.12, stop_words=number_stopwords)
tfidf_vectors = vectorizer.fit_transform(opt_sents)

tra_vectors = vectorizer.transform(tra_sents)
test_vectors = vectorizer.transform(test_sents)


opt_results = {}
opt_results_path = os.path.join(main_folder, 'Results/results_070919_tfidf_only_mindf_001_maxdf_0_12_numberstoplwords.pickle')
results_path = os.path.join(main_folder, 'Results/scores_tfidf_only_mindf_001_maxdf_0_12_numberstoplwords.txt')

with open(results_path, 'w') as f:
    f.write('\n-----------------------------------\n')

### Decision Tree
classifier = DecisionTreeClassifier(criterion='gini')
params = {
    'max_depth': [None] + [*range(15, 35, 5)],
    'min_samples_split': [*range(50, 200, 20)],
    'min_samples_leaf': [*range(3, 14, 2)],
    'max_features': [None, 'sqrt', 'log2']
}

dt_clf = GridSearchCV(classifier, params, cv=5, refit=False, scoring=scoring, n_jobs=n_jobs)
dt_clf = dt_clf.fit(tfidf_vectors, y_opt)
    
with open(results_path, 'a') as file_:
    file_.write('Decision Tree\n')
    file_.write('\nBest Score\n')
    file_.write(str(dt_clf.best_score_))
    file_.write('\nBest Params\n')
    file_.write(str(dt_clf.best_params_))

DT = DecisionTreeClassifier(criterion='gini', **dt_clf.best_params_)

DT.fit(tra_vectors, y_tra)

y_pred = DT.predict(test_vectors)

opt_results['DecisionTree'] = {}
opt_results['DecisionTree']['GridSearchCV'] = dt_clf
opt_results['DecisionTree']['classif_report'] = classification_report(y_test, y_pred)

with open(results_path, 'a') as f:
    f.write('\nClassification Report:\n')
    f.write(classification_report(y_test, y_pred))
    f.write('\nScikit_F1_Macro: ' + str(scikit_f1_score(y_test, y_pred, average='macro')))

with open(opt_results_path, 'wb') as file_:
    pickle.dump(opt_results, file_, protocol=pickle.HIGHEST_PROTOCOL)

    
    
### RandomForestClassifier
classifier = RandomForestClassifier(criterion='gini')
params = {
    'n_estimators': [50] + [*range(100, 1000, 100)], 
    'max_depth': [None] + [*range(65, 115, 15)], 
    'min_samples_split': [10, 20, 30, 40, 50, 100, 120, 150],
    'max_features': ['sqrt', 'log2', 0.1, 0.2],
    'bootstrap': [True, False]
}

rf_clf = GridSearchCV(classifier, params, cv=5, refit=False, scoring=scoring, n_jobs=n_jobs)
rf_clf = rf_clf.fit(tfidf_vectors, y_opt)

with open(results_path, 'a') as f:
    f.write('\n-----------------------------------\n')
    f.write('Random Forest\n')
    f.write('\nBest Score\n')
    f.write(str(rf_clf.best_score_))
    f.write('\nBest Params\n')
    f.write(str(rf_clf.best_params_))

RF = RandomForestClassifier(criterion='gini', **rf_clf.best_params_, random_state=0)
RF.fit(tra_vectors, y_tra)

y_pred = RF.predict(test_vectors)

opt_results['RandomForest'] = {}
opt_results['RandomForest']['GridSearchCV'] = rf_clf
opt_results['RandomForest']['classif_report'] = classification_report(y_test, y_pred)

with open(results_path, 'a') as f:
    f.write('\nClassification Report:\n')
    f.write(classification_report(y_test, y_pred))
    f.write('\nScikit_F1_Macro: ' + str(scikit_f1_score(y_test, y_pred, average='macro')))

with open(opt_results_path, 'wb') as file_:
    pickle.dump(opt_results, file_, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
### SVC
classifier =  SVC()
params = {
    'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 
    'C': [0.025, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7],
}

svc_clf = GridSearchCV(classifier, params, cv=5, refit=False, scoring=scoring, n_jobs=n_jobs)
svc_clf = svc_clf.fit(tfidf_vectors, y_opt)

with open(results_path, 'a') as f:
    f.write('\n-----------------------------------\n')
    f.write('SVM\n')
    f.write('\nBest Score\n')
    f.write(str(svc_clf.best_score_))
    f.write('\nBest Params\n')
    f.write(str(svc_clf.best_params_))

SV = SVC(**svc_clf.best_params_, random_state=0)

SV.fit(tra_vectors, y_tra)

y_pred = SV.predict(test_vectors)

opt_results['SVC'] = {}
opt_results['SVC']['GridSearchCV'] = svc_clf
opt_results['SVC']['classif_report'] = classification_report(y_test, y_pred)

with open(results_path, 'a') as f:
    f.write('\nClassification Report:\n')
    f.write(classification_report(y_test, y_pred))
    f.write('\nScikit_F1_Macro: ' + str(scikit_f1_score(y_test, y_pred, average='macro')))

with open(opt_results_path, 'wb') as file_:
    pickle.dump(opt_results, file_, protocol=pickle.HIGHEST_PROTOCOL)  
    
###
with open(results_path, 'a') as f:
    f.write('Finished')
    
    
    
    
    
    
    