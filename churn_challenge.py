# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 17:59:35 2018

@author: mattf
"""


##### DATA CHALLENGE FOR WEEK 5 - WHAT DRIVES EMPLOYEE CHURN? #####



### what are the goals? 

#What are the main factors that drive employee churn? Do they make sense? Explain.

#What might you be able to do for the company to address employee Churn, what would be follow-up actions?

#If you could add to this data set just one variable that could help explain employee churn, what would that be?





### import the data

import pandas as pd
import os

os.chdir('C:\\Users\mattf\Desktop\insight\interview_prep\data_challenges\week5')

data = pd.read_csv('employee_retention_data.csv')





### inspect the data

# check features and types
data.dtypes

# remove employee ID
data = data.drop('employee_id', axis=1)

# set company ID and department as category
data.company_id = data.company_id.astype('category')
data.dept = data.dept.astype('category')

# set join date and quit date to date type
data.join_date = pd.to_datetime(data.join_date)
data.quit_date = pd.to_datetime(data.quit_date, errors='coerce') # errors due to NaN for current employees





### feature engineering

import numpy as np

# save a copy with only the base features
dataOrig = data.copy()


# salary normalized by average salary within a comapny, across departments
compSalaryDict = data.groupby('company_id').mean().salary.to_dict()
data['salary_compNorm'] = data['salary'] / data['company_id'].map(compSalaryDict)

# salary normalized by the average salary within department types, across companies
deptSalaryDict = data.groupby('dept').mean().salary.to_dict()
data['salary_deptNorm'] = data['salary'] / data['dept'].map(deptSalaryDict)

# salary normalized by the average company department salary
compDeptSalaryDict = data.groupby(['company_id','dept']).mean().salary.to_dict()
data['salary_compDeptNorm'] = data['salary'] / data.set_index(['company_id', 'dept']).index.map(compDeptSalaryDict)

# salary by the number of years of experience before hire
data['salary_expNorm'] = data.salary / data.seniority

# variable to store if they quit or didnt 
data['quitters'] = np.where(pd.isnull(data.quit_date), 1, 0) # 0 = they quit 
dataOrig['quitters'] = np.where(pd.isnull(dataOrig.quit_date), 1, 0) # 0 = they quit 

# fill in quit_date NaN with date data was collected: 2015/12/13
endDate = pd.to_datetime('2015/12/13')
data.quit_date = data.quit_date.fillna(endDate)

# variable to store how long they worked at the company (units = days)
data['duration'] = (data.quit_date - data.join_date).dt.days

# salary by the number of years they worked at the company
data['salary_durNorm'] = data.salary / data.duration

# variable to mark the year they started
data['join_year'] = data.join_date.dt.year

# and the month
# variable to mark the year they started
data['join_month'] = data.join_date.dt.month




### pre-process the data for training models

## target column
y = dataOrig.quitters

## save copy of data before transforms
df_orig = dataOrig.copy()
df = data.copy()


# import pre-processing methods
import sklearn

from sklearn.preprocessing import StandardScaler
scalerOrig = StandardScaler()
scaler = StandardScaler()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(sparse=False)


## list potential predictive features by type

# original data
orig_num_cols = ['salary', 'seniority']
orig_cat_cols = ['company_id', 'dept']

# feature-engineered data
num_cols = ['salary', 'seniority', 'salary_compNorm', 'salary_deptNorm', 'salary_compDeptNorm',
            'salary_expNorm', 'duration', 'salary_durNorm']
cat_cols = ['company_id', 'dept', 'join_year', 'join_month']


## scale the numberical columns
dataOrig[orig_num_cols] = scalerOrig.fit_transform(dataOrig[orig_num_cols]) # original data
data[num_cols] = scaler.fit_transform(data[num_cols]) # feature-engineered data


## one hot encode the categorical columns

# original data
for col in dataOrig.columns.values:
    for col in orig_cat_cols:
        dle = dataOrig[col]
        le.fit(dle.values)
        dataOrig[col] = le.transform(dataOrig[col])
        
for col in orig_cat_cols:
    dencf = dataOrig[[col]]
    oh.fit(dencf)
    temp = oh.transform(dataOrig[[col]])
    temp = pd.DataFrame(temp,columns=[(col+'_'+str(i)) for i in dencf[col].value_counts().index])
    temp = temp.set_index(dataOrig.index.values)
    dataOrig = pd.concat([dataOrig,temp],axis=1)


# feature-engineered data
for col in data.columns.values:
    for col in cat_cols:
        dle = data[col]
        le.fit(dle.values)
        data[col] = le.transform(data[col])
        
for col in cat_cols:
    dencf = data[[col]]
    oh.fit(dencf)
    temp = oh.transform(data[[col]])
    temp = pd.DataFrame(temp,columns=[(col+'_'+str(i)) for i in dencf[col].value_counts().index])
    temp = temp.set_index(data.index.values)
    data = pd.concat([data,temp],axis=1)
    

## drop redundant / useless features
drop_list_orig = orig_cat_cols + ['join_date', 'quit_date', 'quitters'] 
drop_list = cat_cols + ['join_date', 'quit_date', 'quitters'] 

X_orig = dataOrig.drop(drop_list_orig, axis=1)
X = data.drop(drop_list,axis=1)





### Train models on the two datasets


## split data into train/test subsets
from sklearn.model_selection import train_test_split

X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(X_orig, y, test_size=0.2, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


## fit random forest models to the two datasets
from sklearn.ensemble import RandomForestClassifier

forest_orig = RandomForestClassifier()
forest= RandomForestClassifier()

forest_orig.fit(X_orig_train, y_orig_train)
forest.fit(X_train, y_train)


## make predications on test set, investigate model metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


# function to get model metrics 
def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted)             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted)
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted)
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    
    return accuracy, precision, recall, f1


# function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="black" if cm[i, j] < thresh else "white", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt

# make predictions
y_orig_pred = forest_orig.predict(X_orig_test)
y_pred = forest.predict(X_test)

# get metrics and plot confusion matrix for original data model
accuracy_orig, precision_orig, recall_orig, f1_orig = get_metrics(y_orig_test, y_orig_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_orig, precision_orig, recall_orig, f1_orig))

cm_orig =  confusion_matrix(y_orig_test, y_orig_pred)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm_orig, classes=['Quitters','Employees'], normalize=False, title='Original Data Confusion matrix', cmap=plt.cm.Blues)
plt.show()

# get metrics and plot confusion matrix for engineered data model
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

cm =  confusion_matrix(y_test, y_pred)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Quitters','Employees'], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
plt.show()





### Paramaterize the model

## check parameters of random forest model
from pprint import pprint

pprint(forest.get_params())


## create lists of possible parameter values

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


## inititate a random grid search

from sklearn.model_selection import RandomizedSearchCV

# search across 100 different combinations, and use all available cores
forest_random = RandomizedSearchCV(estimator = forest, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)

# Fit the random search model
forest_random.fit(X_train, y_train)


## check best parameters, use that model
pprint(forest_random.best_params_)
#{'bootstrap': False,
# 'max_depth': 90,
# 'max_features': 'auto',
# 'min_samples_leaf': 1,
# 'min_samples_split': 5,
# 'n_estimators': 400}


best_forest = forest_random.best_estimator_

yb_pred = best_forest.predict(X_test)

accuracy, precision, recall, f1 = get_metrics(y_test, yb_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

cm =  confusion_matrix(y_test, yb_pred)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Quitters','Employees'], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
plt.show()






### Visualize the results of the model

# plot feature importances
importances = best_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

feats = {} 
for feature, importance in zip(X.columns, importances):
    feats[feature] = importance 

impp = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
impp.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', color='r', rot=90, yerr=std[indices], align='center')

plt.title('Feature importances')
plt.xlabel('Feature Name')
plt.ylabel('Relative Importance')
plt.xticks(fontsize=8)



# visualize duration frequencies by quit-class

fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)
fig.tight_layout()

ax1.hist(df.loc[df.quitters==0].duration,bins=int(max(df.duration)/2))
ax1.set_title('Ex-Employees')
ax1.set_xlabel('Days Stayed Before Quitting')
ax1.set_ylabel('Count')

ax2.hist(df.loc[df.quitters==1].duration,bins=int(max(df.duration)/2))
ax2.set_title('Current Employees')
ax2.set_xlabel('Days Stayed To-Date')
ax2.set_ylabel('Count')

axes = plt.gca()
axes.set_ylim([0,400])


# visualize the number of people who quit every year

fig, (ax1) = plt.subplots(1,1)

fig.suptitle('Number of Employees Lost Per Year')

ax1.hist(df.loc[df.quitters==0].quit_date.dt.year)
ax1.set_ylabel('Number Employees Lost')
ax1.set_xlabel('Year')


# visualize the year that all ex-employees and current employees were hired

fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)

fig.suptitle('Year of Hire, Ex- and Current Employees')

ax1.hist(df.loc[df.quitters==0].join_date.dt.year)
ax1.set_ylabel('Count')
ax1.set_xlabel('Year of Hire, Ex-employees')

ax2.hist(df.loc[df.quitters==1].join_date.dt.year)
ax2.set_xlabel('Year of Hire, Current Employees')


# visualize the month that all ex-employees and current employees were hired

fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)

fig.suptitle('Year of Hire, Ex- and Current Employees')

ax1.hist(df.loc[df.quitters==0].join_date.dt.month)
ax1.set_ylabel('Count')
ax1.set_xlabel('Month of Hire, Ex-employees')

ax2.hist(df.loc[df.quitters==1].join_date.dt.month)
ax2.set_xlabel('Month of Hire, Current Employees')

## visualize the duration-normalized salary by quit-class
#
#fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)
#
#fig.suptitle('Duration-Normalized Salary for Ex and Current Employees')
#
#ax1.boxplot(df.loc[df.quitters==0].salary_durNorm)
#ax1.set_ylabel('Duration-Normalized Salary')
#ax1.set_xlabel('Ex-Employees')
#
#ax2.boxplot(df.loc[df.quitters==1].salary_durNorm)
#ax2.set_xlabel('Current Employees')


# visualize the absolute salary by quit-class

fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)

fig.suptitle('Average Salary During Employment for Ex and Current Employees')

ax1.boxplot(df.loc[df.quitters==0].salary)
ax1.set_ylabel('Average Salary')
ax1.set_xlabel('Ex-Employees')

ax2.boxplot(df.loc[df.quitters==1].salary)
ax2.set_xlabel('Current Employees')



# what are salary differences in the first year employees

newbies = df.loc[df.duration <=365]

fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)

fig.suptitle('Average Salary During First Year of Employment for Ex and Current Employees')

ax1.boxplot(newbies.loc[newbies.quitters==0].salary_compNorm)
ax1.set_ylabel('Average Salary, Normalized per Company')
ax1.set_xlabel('Ex-Employees, Quit in first year')

ax2.boxplot(df.loc[df.quitters==1].salary_compNorm)
ax2.set_xlabel('Current Employees, Employed < 1 year')





### Linear Regression for employee duration 

ylr = df.duration
lr_drop_cols =['duration', 'salary_durNorm']
Xlr = X.drop(lr_drop_cols, axis=1)

Xlr_train, Xlr_test, ylr_train, ylr_test = train_test_split(Xlr, ylr, test_size=0.2)


## fit the model
from sklearn.linear_model import LinearRegression

lr = LinearRegression(normalize=False)

lr.fit(Xlr_train,ylr_train,sample_weight=None)


## evaluate the predications

lr_pred = lr.predict(Xlr_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

print('\nMean absolute error: %.2f' % mean_absolute_error(ylr_test, lr_pred),'days')

print('\nVariance score: %.2f' % r2_score(ylr_test, lr_pred))


# Check which coefficients are important
print('Coefficients: \n', lr.coef_)



# plot the regression coefficients

coef_df = pd.DataFrame({'coef': lr.coef_,
                        'varname': list(Xlr)
                       })

fig, ax = plt.subplots(figsize=(12, 8))
coef_df.plot(x='varname', y='coef', kind='bar', 
             ax=ax, color='none', legend=False)
ax.set_ylabel('')
ax.set_xlabel('')
ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
           marker='s', s=120, 
           y=coef_df['coef'], color='black')
ax.axhline(y=0, linestyle='--', color='black', linewidth=4)
ax.xaxis.set_ticks_position('none')
_ = ax.set_xticklabels(list(Xlr), 
                       rotation=45, fontsize=8)














