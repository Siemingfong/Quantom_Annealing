# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


def get_model(model, clf_params):
    if model == 'lr':
        clf = LogisticRegression(**clf_params)
    elif model == 'p':
        clf = Perceptron(**clf_params)
    elif model == 'ab':
        clf = AdaBoostClassifier(**clf_params)
    elif model == 'rf':
        clf = RandomForestClassifier(**clf_params)
    elif model == 'svm':
        clf = SVC(**clf_params)
    elif model == 'xgb':
        clf = xgb.XGBClassifier(**clf_params)
    elif model == 'lgb':
        clf = lgb.LGBMClassifier(**clf_params)
    return clf

def get_params(model):
    if model == 'lr':
        clf_params = {
        }
    elif model == 'p':
        clf_params = {
        }
    elif model == 'ab':
        clf_params = {
            'n_estimators': 1000
        }
    elif model == 'rf':
        clf_params = {
            'n_estimators': 1000
        }
    elif model == 'svm':
        clf_params = {
        }
    elif model == 'xgb':
        clf_params = {
            'n_estimators': 1000,
            'importance_type': 'gain'
        }
    elif model == 'lgb':
        clf_params = {
            'n_estimators': 1000,
            'num_leaves': 51,
            'importance_type': 'gain'
        }
    return clf_params