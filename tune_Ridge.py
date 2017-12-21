# import package
import numpy as np
import pandas as pd
import gc
import math

from sklearn.linear_model import Ridge
import lightgbm as lgb

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.sparse import csr_matrix, hstack


# define constant
NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000

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


def main():
    
    # step1:load data
    print("step1:load data")
    
    # load data
    train = pd.read_table('./data/train.tsv', delimiter='\t')
    test = pd.read_table('./data/test.tsv', delimiter='\t')
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

    # category_name & brand_name
    cutting(merge)
    print('cut `category_name&brand_name` completed')
    
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
                         ngram_range=(1, 3),
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

    # Ridge
    rmse  = make_scorer(RMSE, greater_is_better=False)
    params = [{'copy_X':[True],
                'fit_intercept':[True],
                'normalize':[False],
                'random_state':[101],
                'solver':['auto'],
                'alpha': [1.0,  3.0],
                'max_iter':[50],
                'tol':[0.03,  0.1] 
                }]

    gsReg = GridSearchCV(Ridge(), param_grid=params, cv=3, scoring=rmse, n_jobs=4, verbose = 1)
    
    gsReg.fit(X, y)
    print(gsReg.grid_scores_)
    print('best params is: ', gsReg.best_params_)
    print("best rmse is：", gsReg.best_score_)
    
if __name__ == '__main__':
    main()
