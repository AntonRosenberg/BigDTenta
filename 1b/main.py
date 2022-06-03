import tarfile
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange


def get_label(label):
    if label==1:
        return 'Dog'
    return 'Cat'

def histogram(wrong_list):
    histo = np.zeros(198)
    for arr in wrong_list:
        for element in arr:
            histo[element] += 1
    return histo


def get_scores(array, start):
    sc = []
    for arr in array:
        sc.append(arr[start])
    sc = sum(sc) / len(sc)
    return sc


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
        # plt.title(f'label = {label[label.index==errors]}')
        plt.imshow(pic)
        plt.show()

def plot_feat(feat_list, method):
    plt.figure()
    pic1 = np.array(data.iloc[10])
    plt.title(f'label = {get_label(np.array(label.iloc[10]))}, '+method)
    plt.imshow(pic1.reshape(64, 64).T)
    image = np.zeros(np.shape(data)[1])
    for feat in feat_list:
        image[int(feat)]+=1
    print(image)
    plt.imshow(image.reshape(64,64).T, alpha=0.5)


def count(feat_list, count_list: np.ndarray, ind: int) -> np.ndarray:
    for num in feat_list:
        count_list[ind, num] += 1
    return count_list


if __name__ == '__main__':
    try:
        data = pd.read_csv("/Users/antonrosenberg/Documents/GitHub/BigDTenta/CATSnDOGS.csv") / 255
        label = pd.read_csv("/Users/antonrosenberg/Documents/GitHub/BigDTenta/Labels.csv")
    except:
        data = pd.read_csv("C:/Users/anton\OneDrive\Dokument\GitHub\BigDTenta/CATSnDOGS.csv") / 255
        label = pd.read_csv("C:/Users/anton\OneDrive\Dokument\GitHub\BigDTenta/Labels.csv")

    np_data = np.array(data)
    np_labels = np.array(label).flatten()

    num_runs = 100
    threshold = 0.5

    method=['Logistic Regression', 'Random Forest', 'Linear svc']

    count_feat = np.zeros([3, np.shape(data)[1]])
    for i in trange(num_runs):
        X_boot, y_boot = resample(data, label, n_samples=round(len(data)))

        LogReg = LogisticRegression(C=0.05736152510448681,max_iter=1000)
        LogReg.fit(X_boot, y_boot.values.ravel())
        model = SelectFromModel(LogReg, prefit=True)
        LogReg_features = model.get_support(indices=True)
        X_LogReg = model.transform(X_boot)

        count(feat_list=LogReg_features, count_list=count_feat, ind=0)

        RandForest = RandomForestClassifier()
        RandForest.fit(X_boot, y_boot.values.ravel())
        model = SelectFromModel(RandForest, prefit=True)
        RandForest_features = model.get_support(indices=True)
        X_RandForest = model.transform(X_boot)

        count(feat_list=RandForest_features, count_list=count_feat, ind=1)

        lsvc = SVC(C=9, kernel='linear', max_iter=10000)
        lsvc.fit(X_boot, y_boot.values.ravel())
        model = SelectFromModel(lsvc, prefit=True)
        lsvc_features = model.get_support(indices=True)
        X_lsvc = model.transform(X_boot)

        count(feat_list=lsvc_features, count_list=count_feat, ind=2)

        print(len(LogReg_features), np.shape(X_LogReg))
        print(len(RandForest_features), np.shape(X_RandForest))
        print(len(lsvc_features), np.shape(X_lsvc))

    plt.figure()
    plt.title('Logistic Regression')
    plt.bar(range(np.shape(data)[1]), count_feat[0, :])
    plt.figure()
    plt.title('Random Forest')
    plt.bar(range(np.shape(data)[1]), count_feat[1, :])
    plt.figure()
    plt.title('Linear svc')
    plt.bar(range(np.shape(data)[1]), count_feat[2, :])

    count_feat[count_feat < num_runs*threshold] = 0

    for i in range(len(count_feat[:,0])):
        feat_list = np.nonzero(count_feat[i, :])
        plot_feat(feat_list[0], method[i])

    plt.show()

