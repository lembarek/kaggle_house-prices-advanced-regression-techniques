##
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import reciprocal, uniform
from decimal import Decimal
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from bibs.ModelHelper import ModelHelper
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np


##
n_folds=2
def rmsle_cv(model, X, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X)
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attributes_names]

def get_columns(types):
    columns=[]
    for t in types:
        cs = list(data.dtypes[data.dtypes == t].keys())
        columns = columns+cs
    return columns

##
data = pd.read_csv('train.csv')

predict_data = pd.read_csv('test.csv')

predict_data['SalePrice'] = -1

all_data = data.append(predict_data)

num_columns = get_columns(['float64', 'int64'])
num_columns.remove('Id')
num_columns.remove('SalePrice')

cat_columns = get_columns(['O'])

all_data[cat_columns] = all_data[cat_columns].fillna('Autre')

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(num_columns)),
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(cat_columns)),
    ("cat_encoder", OneHotEncoder(sparse=False))
])

preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

##
all_data = preprocess_pipeline.fit_transform(all_data)

X_train = all_data[:1460]

y_train = data['SalePrice']

##
svr_params={
    'gamma': 'scale',
    'kernel': 'linear'
}

rf_params={
    'n_estimators': 100
}

ada_params={
    'n_estimators': 100
}


extra_params={
    'n_estimators': 100
}

xgb_params={
'colsample_bytree' :0.4603,
'gamma' :0.0468,
'learning_rate' :0.05,
'max_depth' :3,
'min_child_weight' :1.7817,
'n_estimators' :2200,
'reg_alpha' :0.4640,
'reg_lambda' :0.8571,
'subsample' :0.5213,
'silent' :1,
'random_state' :7,
'nthread' : -1,
}

lgb_params = {
'objective' : 'regression',
'num_leaves' : 5,
'learning_rate' : 0.05,
'n_estimators' : 720,
'max_bin' : 55,
'sample_fraction' : 0.8,
'sample_freq' : 5,
'feature_fraction' : 0.2319,
'feature_fraction_seed' : 9,
'bagging_seed' : 9,
'min_data_in_leaf' : 6,
'min_sum_hessian_in_leaf' : 11,
'verbose': -1,
    }


params = {}

svr_h = ModelHelper(model=SVR, params=svr_params)
rf_h = ModelHelper(model=RandomForestRegressor, params=rf_params)
ada_h = ModelHelper(model=AdaBoostRegressor, params=ada_params)
extra_h = ModelHelper(model=ExtraTreesRegressor, params=extra_params)
xgb_h = ModelHelper(model=xgb.XGBRegressor, params=xgb_params)
lgb_h = ModelHelper(model=lgb.LGBMRegressor, params=lgb_params)


models = [
    lgb_h.model,
    rf_h.model,
    ada_h.model,
    # svr_h.model,
    extra_h.model,
    xgb_h.model,
    ]
estimators = [(m.__class__.__name__, m) for m in models]

voting_reg = ModelHelper(model=VotingRegressor, params={
  'estimators': estimators
})
models.append(voting_reg.model)



##
for m in models:
  X_train_few, X_test_few, y_train_few, y_test_few = train_test_split(X_train, y_train, test_size=0.1)
  m.fit(X_train_few, y_train_few)
  print(m.__class__.__name__, end=':')
  score = rmsle_cv(m, X_test_few, y_test_few)
  print("{:.4f}".format(score.mean()))


##
grid_params = {
        'scoring':'neg_mean_squared_error',
        'return_train_score':True,
     }

##
params_grid_svr = {
      'kernel': ['rbf', 'linear', 'sigmoid' ],
      'gamma': ['scale', 'auto'],
      'degree': range(1, 10),
      'C': range(1, 5),
      'epsilon': uniform(0, 1)
    };

svr_h.find_best_model(X_train, y_train,
    model_params = params_grid_svr,
    grid_params = grid_params  )


##
params_grid_rf = {
    'n_estimators': [101],
    'max_depth':  [54],
};

rf_h.find_best_model(X_train, y_train,
    model_params = params_grid_rf,
    grid_params = grid_params  )


##
params_grid_xgb = {
    'n_estimators': [50, 150],
    'max_depth':  [50, 100],
    'gamma':[i/10.0 for i in range(3,6)],
    'min_child_weight':[4,5],
    'gamma':[i/10.0 for i in range(3,6)],
    'subsample':[i/10.0 for i in range(6,11)],
    'colsample_bytree':[i/10.0 for i in range(6,11)],
};

xgb_h.find_best_model(X_train, y_train,
    model_params = params_grid_xgb,
    grid_params = grid_params  )

##
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_train, grid_search.predict(X_train)))
print(grid_search.best_params_)


##
X_pred = all_data[1460:]

##
clf = voting_reg.model

y_pred = clf.predict(X_pred)
res = pd.DataFrame()

res['Id'] = predict_data['Id']
res['SalePrice'] = y_pred

res.to_csv('submission_v2.csv', index=False)
