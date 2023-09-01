#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
geo_level_1_id, geo_level_2_id, geo_level_3_id (type: int): geographic region in which building exists, from largest (level 1) to most specific sub-region (level 3). Possible values: level 1: 0-30, level 2: 0-1427, level 3: 0-12567.

(DROPPED) count_floors_pre_eq (type: int): number of floors in the building before the earthquake.

age (type: int): age of the building in years.

area_percentage (type: int): normalized area of the building footprint.

height_percentage (type: int): normalized height of the building footprint.

(OHE) land_surface_condition (type: categorical): surface condition of the land where the building was built. Possible values: n, o, t.

(OHE) foundation_type (type: categorical): type of foundation used while building. Possible values: h, i, r, u, w.

(OHE) roof_type (type: categorical): type of roof used while building. Possible values: n, q, x.

(OHE) ground_floor_type (type: categorical): type of the ground floor. Possible values: f, m, v, x, z.

(OHE) other_floor_type (type: categorical): type of constructions used in higher than the ground floors (except of roof). Possible values: j, q, s, x.

(OHE) position (type: categorical): position of the building. Possible values: j, o, s, t.

(OHE) plan_configuration (type: categorical): building plan configuration. Possible values: a, c, d, f, m, n, o, q, s, u.

(BINARY) has_superstructure_adobe_mud (type: binary): flag variable that indicates if the superstructure was made of Adobe/Mud.

(BINARY) has_superstructure_mud_mortar_stone (type: binary): flag variable that indicates if the superstructure was made of Mud Mortar - Stone.

(BINARY) has_superstructure_stone_flag (type: binary): flag variable that indicates if the superstructure was made of Stone.

(BINARY) has_superstructure_cement_mortar_stone (type: binary): flag variable that indicates if the superstructure was made of Cement Mortar - Stone.

(BINARY) has_superstructure_mud_mortar_brick (type: binary): flag variable that indicates if the superstructure was made of Mud Mortar - Brick.

(BINARY) has_superstructure_cement_mortar_brick (type: binary): flag variable that indicates if the superstructure was made of Cement Mortar - Brick.

(BINARY) has_superstructure_timber (type: binary): flag variable that indicates if the superstructure was made of Timber.

(BINARY) has_superstructure_bamboo (type: binary): flag variable that indicates if the superstructure was made of Bamboo.

(BINARY) has_superstructure_rc_non_engineered (type: binary): flag variable that indicates if the superstructure was made of non-engineered reinforced concrete.

(BINARY) has_superstructure_rc_engineered (type: binary): flag variable that indicates if the superstructure was made of engineered reinforced concrete.

(BINARY) has_superstructure_other (type: binary): flag variable that indicates if the superstructure was made of any other material.

(DROPPED) legal_ownership_status (type: categorical): legal ownership status of the land where building was built. Possible values: a, r, v, w.

count_families (type: int): number of families that live in the building.

(BINARY) has_secondary_use (type: binary): flag variable that indicates if the building was used for any secondary purpose.

(BINARY) has_secondary_use_agriculture (type: binary): flag variable that indicates if the building was used for agricultural purposes.

(BINARY) has_secondary_use_hotel (type: binary): flag variable that indicates if the building was used as a hotel.

(BINARY) has_secondary_use_rental (type: binary): flag variable that indicates if the building was used for rental purposes.

(BINARY) has_secondary_use_institution (type: binary): flag variable that indicates if the building was used as a location of any institution.

(BINARY) has_secondary_use_school (type: binary): flag variable that indicates if the building was used as a school.

(BINARY) has_secondary_use_industry (type: binary): flag variable that indicates if the building was used for industrial purposes.

(BINARY) has_secondary_use_health_post (type: binary): flag variable that indicates if the building was used as a health post.

(BINARY) has_secondary_use_gov_office (type: binary): flag variable that indicates if the building was used fas a government office.

(BINARY) has_secondary_use_use_police (type: binary): flag variable that indicates if the building was used as a police station.

(BINARY) has_secondary_use_other (type: binary): flag variable that indicates if the building was secondarily used for other purposes.
"""

import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_selection import chi2

train_values = pd.read_csv('Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Values.csv')
train_labels = pd.read_csv('Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Labels.csv')
test_values = pd.read_csv('Richters_Predictor_Modeling_Earthquake_Damage_-_Test_Values.csv')

sns.histplot(test_values['age'], bins=35)
scaler = StandardScaler()
train_values['age'] = scaler.fit_transform(np.array(train_values['age']).reshape(-1,1))
#sns.histplot(train_values['age'], bins=35)
building_id = test_values['building_id']

secondary = ['has_secondary_use_agriculture', 'has_secondary_use_hotel',
            'has_secondary_use_rental', 'has_secondary_use_institution',
            'has_secondary_use_school', 'has_secondary_use_industry',
            'has_secondary_use_health_post', 'has_secondary_use_gov_office',
            'has_secondary_use_use_police', 'has_secondary_use_other',
            'has_secondary_use']

binary_values = ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
                'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
                'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
                'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_other']

categorical = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
               'other_floor_type', 'position', 'plan_configuration']

drop_columns = ['building_id', 'legal_ownership_status', 'count_floors_pre_eq', 'has_secondary_use_use_police']
train_values = train_values.drop(drop_columns, axis=1)

#SCALING THE AGE (EXPERIMENTAL)
scaler = MinMaxScaler()
train_values['age'] = scaler.fit_transform(np.array(train_values['age']).reshape(-1,1))
test_values['age'] = scaler.fit_transform(np.array(test_values['age']).reshape(-1,1))

#Concat Area (EXPERIMENTAL)
train_values['plan_configuration'] = ['d' if plan in ['s', 'c', 'a', 'o', 'm', 'n', 'f'] else plan for plan in train_values['plan_configuration']]
train_values['foundation_type'] = ['i' if foundation == 'h' else foundation for foundation in train_values['foundation_type']]

train_values['height_percentage'] = [height/100 for height in train_values['height_percentage']]
train_values['area_percentage'] = [area/100 for area in train_values['area_percentage']]

sns.histplot(train_values['area_percentage'], bins=40)

categorical_df = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
                  'other_floor_type', 'position', 'plan_configuration']

ohe = OneHotEncoder(sparse_output=False)   
le = LabelEncoder() 
encoded_dataframe = pd.DataFrame()
for i in categorical_df:
    temporary = pd.DataFrame(ohe.fit_transform(np.array(train_values[i]).reshape(-1, 1)))
    temporary.columns = ohe.get_feature_names_out([i])
    encoded_dataframe = pd.concat([encoded_dataframe, temporary], axis=1)
    train_values.drop([i], axis=1, inplace=True)   
    
x = pd.concat([train_values, encoded_dataframe], axis=1)
y = le.fit_transform(train_labels['damage_grade'])

#Finding the correlation between two columns. If it's higher than 70%, then one of them will be dropped.    
def find_high_correlation_columns(df):
    le = LabelEncoder()
    for i in df.columns:
        for j in df.columns:
            if i == j:
                pass
            else:
                if (type(df[i][1]) != str) and (type(df[j][1]) != str):
                    if df[i].corr(df[j])*100 > 70:
                        print(i, j, df[i].corr(df[j])*100)
                elif type(df[i][1]) != str and (type(df[j][1]) == str):
                    if df[i].corr(pd.DataFrame(le.fit_transform(df[j]))[0])*100 > 70:
                        print(i, j, df[i].corr(pd.DataFrame(le.fit_transform(df[j]))[0])*100)
                elif type(df[i][1]) == str and (type(df[j][1]) != str):
                    if df[j].corr(pd.DataFrame(le.fit_transform(df[i]))[0])*100 > 70:
                        print(i, j, df[j].corr(pd.DataFrame(le.fit_transform(df[i]))[0])*100)
                else:
                    if pd.DataFrame(le.fit_transform(df[j]))[0].corr(pd.DataFrame(le.fit_transform(df[i]))[0])*100 > 70:
                        print(i, j, pd.DataFrame(le.fit_transform(df[j]))[0].corr(pd.DataFrame(le.fit_transform(df[i]))[0])*100)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

test_values = test_values.drop(drop_columns, axis=1)

#Concat Area (EXPERIMENTAL)
test_values['plan_configuration'] = ['d' if plan in ['s', 'c', 'a', 'o', 'm', 'n', 'f'] else plan for plan in test_values['plan_configuration']]
test_values['foundation_type'] = ['i' if foundation == 'h' else foundation for foundation in test_values['foundation_type']]

test_values['height_percentage'] = [height/100 for height in test_values['height_percentage']]
test_values['area_percentage'] = [area/100 for area in test_values['area_percentage']]

ohe_ = OneHotEncoder(sparse_output=False)   

encoded_test_dataframe = pd.DataFrame()

for i in categorical_df:
    temporary_test = pd.DataFrame(ohe_.fit_transform(np.array(test_values[i]).reshape(-1, 1)))
    temporary_test.columns = ohe_.get_feature_names_out([i])
    encoded_test_dataframe = pd.concat([encoded_test_dataframe, temporary_test], axis=1)
    test_values.drop([i], axis=1, inplace=True)   
    
p = pd.concat([test_values, encoded_test_dataframe], axis=1)

params = [
    {'alpha':[1e-5, 1e-2, 0.05, 0.1, 0.5, 1]}
    ]

xgb = XGBClassifier(objective='multi:softmax', gamma=0, alpha=0.1, max_depth=10,
                    colsample_bytree=0.8, subsample=0.8)
"""
#Getting an alpha:

clf = GridSearchCV(estimator=xgb, param_grid=params, scoring='f1_micro', cv=2, verbose=3)
clf_fit = clf.fit(X_train, y_train)
print(clf_fit.best_estimator_)
print(clf_fit.best_params_)
"""

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
score = f1_score(y_test, y_pred, average='micro')
print(score)

"""
#SUBMIT:

y_pred = xgb.predict(p)
my_list = [int(x) for x in y_pred]

submit = pd.DataFrame(my_list, columns=['damage_grade'])
submit += 1 #THAT IS BECAUSE OF THE PREVIOUS LABEL ENCODING STEP (y)
submit_that = pd.concat([building_id, submit], axis=1)
#submit_that['damage_grade'] = submit_that['damage_grade'].astype(int)
submit_that.to_csv('submit.csv', index=False) 
"""