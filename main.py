import tarfile
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt


def check_predictions(predictions, labels):
    wrong_pred = []
    for ind, pred, label in zip(labels.index, predictions, labels.values.ravel()):
        if pred != label:
            wrong_pred.append(ind)
    return wrong_pred


def plot_err(errors, df):
    pics = [df[df.index == error] for error in errors]
    print(errors)
    pics = [np.array(pic).reshape(64, 64).T for pic in pics]
    for pic in pics:
        plt.title(f'label = {label[label.index==error]}')
        plt.imshow(pic)
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv("CATSnDOGS.csv")
    label = pd.read_csv("Labels.csv")
    np_data = np.array(data)
    np_labels = np.array(label).flatten()

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=8)
    ones = np.size(label.index[label.index == 1])
    zeroes = np.size(label.index[label.index == 0])
    models = [SVC(), RandomForestClassifier(), LogisticRegressionCV(cv=5, max_iter=1000)]
    scores = [cross_val_score(model, data, label.values.ravel(), cv=5, scoring='accuracy') for model in models]

    print(scores)

    wrong_list = [check_predictions(model.fit(x_train, y_train).predict(x_test), y_test) for model in models]
    print(wrong_list)

    plot_err(wrong_list[0], data)
    '''
    pic1 = np.array(data.iloc[0])
    plt.title(f'label = {np.array(label.iloc[0])}')
    plt.imshow(pic1.reshape(64, 64).T)
    '''


