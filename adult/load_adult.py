import pickle, os

import numpy as np
import pandas as pd

from collections import OrderedDict

"""
This script loads and preprocesses the Adult data from UCI ML repository
"""


orig_data_description = OrderedDict({
        "age": "continuous",
        "workclass": "Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked",
        "fnlwgt": "continuous",
        "education": "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool",
        "education-num": "continuous",
        "marital-status": "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse",
        "occupation": "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces",
        "relationship": "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried",
        "race": "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black",
        "sex": "Female, Male",
        "capital-gain": "continuous",
        "capital-loss": "continuous",
        "hours-per-week": "continuous",
        "native-country": "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands",
        "income": "<=50K, >50K"
})

def load_orig_data():
    if os.path.exists("./data/adult_train.csv") and os.path.exists("./data/adult_test.csv"):
        train_data = pd.read_csv("./data/adult_train.csv", index_col=0)
        test_data = pd.read_csv("./data/adult_test.csv", index_col=0)

    else:
        base_url = lambda s: f"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.{s}"

        train_data = pd.read_csv(base_url("data"), header=None, names=orig_data_description.keys(), na_values="?", skipinitialspace=True)
        test_data = pd.read_csv(base_url("test"), header=None, names=orig_data_description.keys(), na_values="?", skipinitialspace=True)

        ## drop na
        train_data = train_data.dropna()
        test_data = test_data.dropna()

        ## for some reason, there is a "." in the end of every line in the test data, so remove that
        test_data["income"] = test_data["income"].map({"<=50K.": "<=50K", ">50K.": ">50K"})
        if os.path.exists("./data/"):
            train_data.to_csv("./data/adult_train.csv")
            test_data.to_csv("./data/adult_test.csv")

    return train_data.copy(), test_data.copy()

def preprocess_adult():
    train_data, test_data = load_orig_data()
    data_description = orig_data_description.copy()
    ## remove the "education-num" feature as it is one-to-one with the "education"
    del train_data["education-num"]
    del test_data["education-num"]
    del data_description["education-num"]


    def onehot_encode(X, encoding=None):
        if encoding is None:
            encoded_X = X.astype("category").cat.codes
            encoding = {cat: code for cat, code in zip(X, encoded_X)}
        else:
            encoded_X = X.map(encoding)
        num_cat = len(encoding)
        diag_matrix = np.eye(num_cat)
        onehotted_X = diag_matrix[encoded_X]
        return onehotted_X, encoding


    ## Encode categorical variables
    preprocessed_train_data = pd.DataFrame()
    preprocessed_test_data = pd.DataFrame()

    encodings = {}

    for key, value in data_description.items():
        if value != "continuous" and key!="income":
            onehotted_feature, encoding = onehot_encode(train_data[key])
            sorted_cats = sorted(encoding, key=encoding.get)
            # convert onehotted numpy frame to pandas with category names appended to the feature
            df_names = [f"{key}:{name}" for name in sorted_cats]
            df = pd.DataFrame(onehotted_feature, columns=df_names)
            # add to preprocessed train data
            preprocessed_train_data = pd.concat((preprocessed_train_data, df), axis=1)
            encodings[key] = encoding
            ## for test
            onehotted_feature_test, _ = onehot_encode(test_data[key], encoding=encoding)
            df_test = pd.DataFrame(onehotted_feature_test, columns=df_names)
            preprocessed_test_data = pd.concat((preprocessed_test_data, df_test), axis=1)
        else:
            preprocessed_train_data[key] = train_data[key].values
            preprocessed_test_data[key] = test_data[key].values


    ## z-normalize the continuous features
    from sklearn.preprocessing import scale

    for key, value in data_description.items():
        if value == "continuous":
            preprocessed_train_data[key] = scale(preprocessed_train_data[key])
            preprocessed_test_data[key] = scale(preprocessed_test_data[key])

    ## label targets
    target_map = {"<=50K": 0, ">50K": 1}
    preprocessed_train_data["income"] = preprocessed_train_data["income"].map(target_map)
    preprocessed_test_data["income"] = preprocessed_test_data["income"].map(target_map)

    encodings["income"] = target_map

    return preprocessed_train_data, preprocessed_test_data, encodings, data_description
