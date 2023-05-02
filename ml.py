import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

import xgboost as xgb

import tensorflow as tf


def separate_prediction(df: pd.DataFrame, model):
    x_data = []
    y = []
    agents = []
    map_name = []
    for index, row in df.iterrows():
        if index % 10 == 0:
            x_row = [row["map"]]
            map_name.append(row["map"])
            y_row = 1 if row["team1-score"] > row["team2-score"] else 0
        x_row.append(row["agent"])
        agents.append(row["agent"])
        if index % 10 == 9:
            x_data.append(x_row)
            y.append(y_row)

    print(set(agents))
    print(len(set(agents)))
    print(set(map_name))
    print(len(set(map_name)))

    # x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.3, random_state=42)
    # x_train_sep = []
    # y_train_sep = []
    # for i in range(len(x_train)):
    #     x_train_sep.append(x_train[i][0:6])
    #     x_train_sep.append([x_train[i][0]] + x_train[i][5:10])
    #     if y_train[i] == 1:
    #         y_train_sep.append(1)
    #         y_train_sep.append(0)
    #     else:
    #         y_train_sep.append(0)
    #         y_train_sep.append(1)


    # x_train_encode = encoder.fit_transform(x_train_sep)
    # print(x_train_encode.shape)
    #
    # model.fit(x_train_encode, y_train_sep)
    #
    # y_pred = []
    # for i in range(len(x_test)):
    #     x_test_1 = x_test[i][0:6]
    #     x_test_2 = [x_test[i][0]] + x_test[i][5:10]
    #     x_test_1_encode = encoder.transform([x_test_1])
    #     x_test_2_encode = encoder.transform([x_test_2])
    #     y_pred_1 = model.predict_proba(x_test_1_encode)[0][1]
    #     y_pred_2 = model.predict_proba(x_test_2_encode)[0][1]
    #     y_pred.append(1 if y_pred_1 >= y_pred_2 else 0)
    #
    # test_accuracy = accuracy_score(y_test, y_pred)
    # test_f1 = f1_score(y_test, y_pred)
    # print(test_accuracy)
    # print(test_f1)

    # use_sklearn(model, x_train_encode, x_test_encode, y_train, y_test)


def use_sklearn(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    print(test_accuracy)
    print(test_f1)


def use_xgboost(x_train, x_test, y_train, y_test):
    classifier = xgb.XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    print(test_accuracy)
    print(test_f1)


def use_tensorflow(x_train, x_test, y_train, y_test, epochs):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs)
    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred_binary)
    test_f1 = f1_score(y_test, y_pred_binary)
    print(test_accuracy)
    print(test_f1)


if __name__ == "__main__":
    df = pd.read_csv("test280.csv")

    # map_x = []
    # team1_x = []
    # team2_x = []
    # team1_acs = []
    # team2_acs = []
    # y = []
    # for index, row in df.iterrows():
    #     if index % 10 == 0:
    #         map_x.append([row["map"]])
    #         team1_row = []
    #         y_row = 1 if row["team1-score"] > row["team2-score"] else 0
    #         team2_row = []
    #         team1_acs_row = []
    #         team2_acs_row = []
    #
    #     if index % 10 < 5:
    #         team1_row.append(row["agent"])
    #         team1_acs_row.append(row["k"])
    #     else:
    #         team2_row.append(row["agent"])
    #         team2_acs_row.append(row["k"])
    #     if index % 10 == 9:
    #         team1_x.append(team1_row)
    #         team2_x.append(team2_row)
    #         team1_acs.append(team1_acs_row)
    #         team2_acs.append(team2_acs_row)
    #         y.append(y_row)
    #
    # map_ = [' '.join(x) for x in map_x]
    # team1_ = [' '.join(x) for x in team1_x]
    # team2_ = [' '.join(x) for x in team2_x]
    #
    # enc = CountVectorizer()
    # team1Agent = (enc.fit_transform(team1_).toarray())
    # enc.get_feature_names_out()
    # team2Agent = (enc.fit_transform(team2_).toarray())
    # enc.get_feature_names_out()
    # maps = (enc.fit_transform(map_).toarray())
    # enc.get_feature_names_out()
    #
    # X = np.hstack([maps, team1Agent, team2Agent, ])

    map_x = []
    team1 = []
    team2 = []
    team1_x = []
    team2_x = []
    y = []
    for index, row in df.iterrows():
        if index % 10 == 0:
            map_x.append([row["map"]])
            team1_row = []
            y_row = 1 if row["team1-score"] > row["team2-score"] else 0
            team2_row = []

        if index % 10 < 5:
            team1_row.append(row["agent"])
        else:
            team2_row.append(row["agent"])

        if index % 10 == 9:
            team1_x.append(team1_row)
            team2_x.append(team2_row)
            team1.append([row['team1']])
            team2.append([row['team2']])
            y.append(y_row)

    map_ = [' '.join(x) for x in map_x]
    team1_ = [' '.join(x) for x in team1_x]
    team2_ = [' '.join(x) for x in team2_x]
    team1Name_ = [' '.join(x) for x in team1]
    team2Name_ = [' '.join(x) for x in team2]

    enc = CountVectorizer()
    team1Agent = (enc.fit_transform(team1_).toarray())
    enc.get_feature_names_out()
    team2Agent = (enc.fit_transform(team2_).toarray())
    enc.get_feature_names_out()
    maps = (enc.fit_transform(map_).toarray())
    enc.get_feature_names_out()
    team1Name = (enc.fit_transform(team1Name_).toarray())
    enc.get_feature_names_out()
    team2Name = (enc.fit_transform(team2Name_).toarray())
    enc.get_feature_names_out()

    X = np.hstack([maps, team1Agent, team2Agent, team1Name, team2Name])

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # use_sklearn(LogisticRegression(), x_train, x_test, y_train, y_test)
    use_tensorflow(x_train, x_test, y_train, y_test, 10)









