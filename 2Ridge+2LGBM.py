# import package
import numpy as np
import pandas as pd
import gc
import math

from sklearn.linear_model import Ridge
import lightgbm as lgb

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix, hstack

import pyximport; pyximport.install()
from joblib import Parallel, delayed


# define constant
NUM_BRANDS = 4500
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 90000

# define function
def RMSE(ground_truth, predictions):
    # define RMSE
    return math.sqrt(mean_squared_error(ground_truth, predictions))
    
def split_cat(text):
    # split category_name
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    
def handle_missing_inplace(dataset):
    # handle missing values of brand_name and item_description
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)

def cutting(dataset):
    # 对brand_name中出现频率没有在前NUM_BRANDS的用‘missing'替代
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    # 同理，对category_name进行相同操作
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'

def to_categorical(dataset):
    # change type of item_condition_id and category_name
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')
    
    
def main():
    
    # step1:load data
    print("step1:load data")
    
    # load data
    train = pd.read_table('../input/train.tsv', delimiter='\t')
    test = pd.read_table('../input/test.tsv', delimiter='\t')
    print('load data completed')
    
    # drop train where price=0
    train.drop(train[(train.price == 0)].index, inplace=True)
    train_len = train.shape[0]
    y = np.log1p(train["price"])
    print('drop train with price=0 completed')
    

    # concat data
    merge = pd.concat([train, test])
    print('concat train and test completed')
    submission = test[['test_id']]


    del train
    del test
    gc.collect()
    
    
    # step2:feature engineering
    print("step2:feature engineering")
    
    # category_name
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    merge.drop('category_name', axis=1, inplace=True)
    print('split `category_name` completed')
    
    # brand_name & item_description
    handle_missing_inplace(merge)
    print('handle missing values of `brand_name&item_description` completed')
    # item_description
    merge['item_description'] =  merge['item_description'] + ' ' +  merge['name']

    # category_name & brand_name
    cutting(merge)
    print('cut `category_name&brand_name` completed')
    
    # item_condition_id & category_name
    to_categorical(merge)
    print('to_categorical `category_name&item_condition_id` completed')
    
    # name
    cv = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = cv.fit_transform(merge['name'])
    print('CountVectorizer `name` completed')
    X_name = X_name.astype('float64')
    
    # category_name
    cv = CountVectorizer()
    X_category1 = cv.fit_transform(merge['general_cat'])
    X_category2 = cv.fit_transform(merge['subcat_1'])
    X_category3 = cv.fit_transform(merge['subcat_2'])
    print('CountVectorizer `category_name` completed')
    
    # item_description
    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 2),
                         stop_words='english')
    X_description = tv.fit_transform(merge['item_description'])
    print('TfidfVectorizer `item_description` completed')
    
    # brand_name
    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('LabelBinarizer `brand_name` completed')
    
    # item_condition_id & shipping
    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],sparse=True).values)
    print('Get dummies on `item_condition_id&shipping` completed')
    

    # combine to create new merge
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()
    print('Create sparse merge completed')
    print('sparse_merge shape is ',sparse_merge.shape)
    
    # select X and X_test from new merge
    X = sparse_merge[:train_len]
    X_test = sparse_merge[train_len:]
    
    
    # step3:modeling
    print("step3:modeling")

    # Ridge1
    train_X3, valid_X3, train_y3, valid_y3 = train_test_split(X, y, test_size = 0.03, random_state = 20) 

    model = Ridge(alpha=.6, copy_X=True, fit_intercept=True, max_iter=100,
      normalize=False, random_state=101, solver='auto', tol=0.01)
    model.fit(train_X3, train_y3)
    
    trainR3 = model.predict(train_X3)
    print('training rmse:', RMSE(train_y3, trainR3))
    validR3 = model.predict(valid_X3)
    print('valid rmse:', RMSE(valid_y3, validR3))

    print('Train ridge1 completed')
    predsR = model.predict(X_test)
    print('Predict ridge1 completed')
    
    # Ridge2
    train_X4, valid_X4, train_y4, valid_y4 = train_test_split(X, y, test_size = 0.03, random_state = 70) 

    model = Ridge(solver='sag', fit_intercept=True, random_state=125)
    model.fit(train_X4, train_y4)
    
    trainR4 = model.predict(train_X4)
    print('training rmse:', RMSE(train_y4, trainR4))
    validR4 = model.predict(valid_X4)
    print('valid rmse:', RMSE(valid_y4, validR4))

    print('Train ridge2 completed')
    predsR2 = model.predict(X_test)
    print('Predict ridge2 completed')

    # lgbm 1
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.1, random_state = 25) 
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(valid_X, label=valid_y)
    watchlist = [d_train, d_valid]
    
    params = {
        'learning_rate': 0.70,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': 4
    }

    model = lgb.train(params, train_set=d_train, num_boost_round=6000, valid_sets=watchlist, \
    early_stopping_rounds=1000, verbose_eval=1000) 
    print('Train lgbm1 completed')
    predsL = model.predict(X_test)
    print('Predict lgbm1 completed')
    
    
    # lgbm2
    train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.1, random_state = 11) 
    d_train2 = lgb.Dataset(train_X2, label=train_y2)
    d_valid2 = lgb.Dataset(valid_X2, label=valid_y2)
    watchlist2 = [d_train2, d_valid2]
    
    params2 = {
        'learning_rate': 0.85,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 2,
        'bagging_fraction': 0.75,
        'nthread': 4
    }
    
    model = lgb.train(params2, train_set=d_train2, num_boost_round=5000, valid_sets=watchlist2, \
    early_stopping_rounds=500, verbose_eval=500) 
    print('Train lgbm2 completed')
    predsL2 = model.predict(X_test)
    print('Predict lgbm2 completed')

    
    # step4: write results to csv
    print("step4:write results to csv")

    preds = predsR*0.15 + predsR2*0.25 + predsL*0.3 + predsL2*0.3

    submission['price'] = np.expm1(preds)
    submission.to_csv("Results_2Ridge_2LGBM.csv", index=False)


if __name__ == '__main__':
    main()