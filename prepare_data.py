# prepare_data for wide deep model input
# include input data x(wide_data, deep_data, target),
# deep_columns_idx={'col_name':index,...}, embedding_columns_dict={'col_name':(unique_value, embedding_dimension)}
# author: WenYi
# time: 2019-09-24
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
import os


def read_data():
    path = os.path.abspath('..')
    data_path = os.path.join(path, 'data', 'adult.data')
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
             'native-country', 'target']
    data = pd.read_table(data_path, sep=',', names=names)
    return data


def feature_engine(data):
    # target
    data['target'] = data['target'].apply(lambda x: 0 if x == ' <=50K' else 1)
    # age
    bins = [-np.inf, 18, 25, 35, 45, 50, np.inf]
    labels = list(range(len(bins) - 1))
    data['age'] = pd.cut(data['age'], bins=bins, labels=labels)

    # education-num
    bins = [-np.inf, 5, 10, 20, 40, np.inf]
    labels = list(range(len(bins) - 1))
    data['education-num'] = pd.cut(data['education-num'], bins=bins, labels=labels)

    # hours-per-week
    bins = [-np.inf, 10, 30, 40, 70, np.inf]
    labels = list(range(len(bins) - 1))
    data['hours-per-week'] = pd.cut(data['hours-per-week'], bins=bins, labels=labels)

    continuous_cols = ['fnlwgt', 'capital-gain', 'capital-loss']
    cat_columns = [col for col in data.columns if
                   col not in continuous_cols + ['age', 'hours-per-week', 'education-num']]

    for col in cat_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    for col in continuous_cols:
        mms = MinMaxScaler()
        data[col] = mms.fit_transform(data[col].values.reshape(-1, 1)).reshape(-1)

    wide_columns = ['age', 'workclass', 'education', 'education-num', 'occupation', 'relationship',
                    'hours-per-week', 'native-country', 'marital-status', 'sex']
    data_wide = data[wide_columns]
    cross_columns = [['occupation', 'sex'], ['occupation', 'education'], ['education', 'native-country'],
                     ['age', 'occupation'], ['age', 'hours-per-week'], ['sex', 'education']]
    for l in cross_columns:
        poly = PolynomialFeatures()
        c = poly.fit_transform(data_wide[l])
        c = pd.DataFrame(c, columns=[l[0] + '_' + l[1] + '_{}'.format(i) for i in range(c.shape[1])])
        data_wide = pd.concat((data_wide, c), axis=1)

    # onehot
    for col in wide_columns:
        data_wide[col] = data_wide[col].astype('str')
    data_wide = pd.get_dummies(data_wide)
    data_target = data['target']

    # 构建embedding dict
    deep_columns = ['workclass', 'occupation', 'native-country', 'race', 'fnlwgt', 'capital-gain', 'capital-loss']
    data_deep = data[deep_columns]
    embedding_columns = ['workclass', 'occupation', 'native-country', 'race']
    embedding_columns_dict = {}
    for i in range(len(deep_columns)):
        if deep_columns[i] in embedding_columns:
            col_name = deep_columns[i]
            embedding_columns_dict[col_name] = (len(data_deep[col_name].unique()), 8)
    deep_columns_idx = dict()
    for idx, key in enumerate(data_deep.columns):
        deep_columns_idx[key] = idx

    train_wide, test_wide = train_test_split(data_wide, test_size=0.4, random_state=999)
    train_deep, test_deep = train_test_split(data_deep, test_size=0.4, random_state=999)
    train_target, test_target = train_test_split(data_target, test_size=0.4, random_state=999)
    train, test = (train_wide, train_deep, train_target), (test_wide, test_deep, test_target)
    return train, test, deep_columns_idx, embedding_columns_dict


if __name__ == "__main__":
    data = read_data()
    train, test, deep_columns_idx, embedding_columns_dict = feature_engine(data)
    print(train[2].values[0])