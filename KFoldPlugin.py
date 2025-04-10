import pandas as pd
import numpy as np
import xgboost
print(xgboost.__file__)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
import warnings
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings('ignore')

import PyPluMA
import PyIO

class KFoldPlugin:
 def input(self, inputfile):
  self.parameters = PyIO.readParameters(inputfile)
 def run(self):
     pass
 def output(self, outputfile):
  stat1_train = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["stat1"], sep="\t")
  origin_train = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["origin"], sep="\t")

  #model = XGBClassifier(n_estimator=1000, max_depth=6, reg_lambda=2, random_state=3, objective='binary:logistic', distribution='bernoulli', scale_pos_weight=0.03, class_weight={0: 1, 1: 33}, eval_metric="logloss")
  model = XGBClassifier(n_estimators=int(self.parameters["nestimators"]), max_depth=6, reg_lambda=2, random_state=3, objective='binary:logistic',  scale_pos_weight=0.03,  eval_metric="logloss")

  model.fit(stat1_train, origin_train)
  kfold = KFold(n_splits=int(self.parameters["nsplit"]), random_state=int(self.parameters["randomstate"]), shuffle=True)

  results = cross_val_score(model, stat1_train, origin_train, cv=kfold)

  print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
  model.save_model(outputfile)

