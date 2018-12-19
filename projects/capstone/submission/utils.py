from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import pandas as pd
import numpy as np

def drop_na_columns(data, label_column='StageName', na_threshold=0.9):
    # drop columns where less than 90% of the rows have values
    row_count, col_count = data.shape
    threshold = int(round(row_count * na_threshold))
    data = data.dropna(thresh=threshold, axis='columns')
    
    # backfill the remaining na cells
    data = data.fillna(method='backfill', axis='columns')
    return data

def drop_rows(data, label_column='StageName', labels=['Closed Lost', 'Closed Won']):
    # Filter down the dataset to the labels we are interested in: Closed Won and Closed Lost
    data = data[data[label_column].isin(labels)]
    return data

from sklearn.model_selection import train_test_split

def data_split(data, labels, test_size=0.2, random_state=42):
    y = labels
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

def encode(data):
    categorical_features = []
    numeric_features = []
    for index, dtype in enumerate(data.dtypes):
        if dtype == 'float64' or dtype =='int64':
            numeric_features.append(data.columns[index])
        else:
            categorical_features.append(data.columns[index])

    encoded_data = encode_onehot(data, categorical_features)
    return encoded_data

def feature_scaler(features):
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features
    

def feature_selection(features, labels, k_features=5):
    test = SelectKBest(score_func=mutual_info_classif, k=k_features) 
    fit = test.fit(features, labels)
    features = fit.transform(features)
    return features

def get_labels(data, label_column='StageName'):
    labels = np.asarray(data[label_column])
    
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y

def drop_corr_columns(data, labels):
    # Create correlation matrix
    corr_matrix = data.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    # Drop features 
    data = data.drop(to_drop, axis='columns')
    
    # Drop features that are correlated with the labels
    l = pd.DataFrame()
    l['labels'] = labels;
    corr = data.corrwith(l)
    cols = list(corr[corr.notnull()].keys())
    data = data.drop(cols, axis='columns')

    return data
    

    