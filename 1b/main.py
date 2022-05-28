import tarfile
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange


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


def count(feat_list, count_list: np.ndarray, ind: int) -> np.ndarray:
    for num in feat_list:
        count_list[ind, num] += 1
    return count_list


if __name__ == '__main__':
    data = pd.read_csv("/Users/antonrosenberg/Documents/GitHub/BigDTenta/CATSnDOGS.csv") / 255
    label = pd.read_csv("/Users/antonrosenberg/Documents/GitHub/BigDTenta/Labels.csv")
    np_data = np.array(data)
    np_labels = np.array(label).flatten()

    num_runs = 100

    count_feat = np.zeros([3, np.shape(data)[1]])
    for i in trange(num_runs):
        X_boot, y_boot = resample(data, label, n_samples=round(len(data)))

        mean_var = np.mean(X_boot.var(axis=1))
        std_var = np.std(X_boot.var(axis=1))
        sel = VarianceThreshold(threshold=(mean_var + std_var))
        sel.fit(X_boot)
        var_features = sel.get_support(indices=True)
        X_var = sel.fit_transform(X_boot)

        count_feat = count(feat_list=var_features, count_list=count_feat, ind=0)

        selK = SelectKBest(chi2, k=500)
        selK.fit(X_boot, y_boot)
        selK_features = selK.get_support(indices=True)
        X_selK = selK.fit_transform(X_boot, y_boot)

        count_feat = count(feat_list=selK_features, count_list=count_feat, ind=1)

        lsvc = LinearSVC(C=500, penalty="l1", dual=False, max_iter=10000)
        lsvc.fit(X_boot, y_boot.values.ravel())
        model = SelectFromModel(lsvc, prefit=True)
        lsvc_features = model.get_support(indices=True)
        X_lsvc = model.transform(X_boot)

        count_feat = count(feat_list=lsvc_features, count_list=count_feat, ind=2)

        print(len(var_features), np.shape(X_var))
        print(len(selK_features), np.shape(X_selK))
        print(len(lsvc_features), np.shape(X_lsvc))
    plt.figure()
    plt.title('Variance filtering')
    plt.bar(range(np.shape(data)[1]), count_feat[0,:])
    plt.figure()
    plt.title('Select K best, chi2 score')
    plt.bar(range(np.shape(data)[1]), count_feat[1, :])
    plt.figure()
    plt.title('Linear svc')
    plt.bar(range(np.shape(data)[1]), count_feat[2, :])
    print(count_feat)

    plt.show()
    '''
    num_runs = 100
    k = 5
    
    for j in trange(num_runs):
        kf = StratifiedKFold(n_splits=k, shuffle=True)
        prediction = []
        models = [SVC(), RandomForestClassifier(), LogisticRegression(max_iter=1000)]
        score = np.zeros([len(models), k])
        fold_ind = 0
        for train_index, test_index in kf.split(x_tr, y_tr):
            X_train, X_test = x_tr.iloc[train_index], x_tr.iloc[test_index]
            y_train, y_test = y_tr.iloc[train_index], y_tr.iloc[test_index]
            for ind, model in enumerate(models):
                model.fit(X_train, y_train.values.ravel())
                prediction = model.predict(X_test)
                score[ind, fold_ind]=(model.score(X_test, y_test))
                tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
                wrong_list.append(check_predictions(prediction, y_test))
                tpr.append(tp/(tp+fn))
                tnr.append(tn/(tn+fp))
            fold_ind += 1
        mean_score.append(np.mean(score, axis=1))
        std.append(np.std(score, axis=1))

    wrong_list_m1 = [wrong_list[i] for i in range(0, len(wrong_list), len(models))]
    wrong_list_m2 = [wrong_list[i] for i in range(1, len(wrong_list), len(models))]
    wrong_list_m3 = [wrong_list[i] for i in range(2, len(wrong_list), len(models))]

    histo1 = histogram(wrong_list_m1)
    histo2 = histogram(wrong_list_m2)
    histo3 = histogram(wrong_list_m3)

    tnr1 = [tnr[i] for i in range(0, len(tnr), len(models))]
    tnr1 = sum(tnr1)/len(tnr1)
    tnr2 = [tnr[i] for i in range(1, len(tnr), len(models))]
    tnr2 = sum(tnr2) / len(tnr2)
    tnr3 = [tnr[i] for i in range(2, len(tnr), len(models))]
    tnr3 = sum(tnr3) / len(tnr3)

    tpr1 = [tpr[i] for i in range(0, len(tpr), len(models))]
    tpr1 = sum(tpr1) / len(tpr1)
    tpr2 = [tpr[i] for i in range(1, len(tpr), len(models))]
    tpr2 = sum(tpr2) / len(tpr2)
    tpr3 = [tpr[i] for i in range(2, len(tpr), len(models))]
    tpr3 = sum(tpr3) / len(tpr3)

    mean_score1 = get_scores(mean_score, 0)
    mean_score2 = get_scores(mean_score, 1)
    mean_score3 = get_scores(mean_score, 2)

    std1 = get_scores(std, 0)
    std2 = get_scores(std, 1)
    std3 = get_scores(std, 2)

    print(f'SVM: accuracy = {mean_score1}, std = {std1}, tpr = {tpr1}, tnr = {tnr1} \n '
          f'RandomForest: accuracy = {mean_score2}, std = {std2}, tpr = {tpr2}, tnr = {tnr2} \n'
          f'LogisticRegression: accuracy = {mean_score3}, std = {std3}, tpr = {tpr3}, tnr = {tnr3}')

    plt.figure()
    plt.title('SVM')
    plt.bar(range(len(histo1)), histo1)
    plt.figure()
    plt.title('RandomForest')
    plt.bar(range(len(histo1)), histo2)
    plt.figure()
    plt.title('LogisticRegression')
    plt.bar(range(len(histo1)), histo3)

    plt.show()
    '''
    '''
    ones = np.size(label.index[label.index == 1])
    zeroes = np.size(label.index[label.index == 0])
    
    scores = [cross_val_score(model, data, label.values.ravel(), cv=5, scoring='accuracy') for model in models]
    
    print(scores)

    wrong_list = [check_predictions(model.fit(x_train, y_train).predict(x_test), y_test) for model in models]
    print(wrong_list)

    plot_err(wrong_list[0], data)
    '''
    '''
    pic1 = np.array(data.iloc[0])
    plt.title(f'label = {np.array(label.iloc[0])}')
    plt.imshow(pic1.reshape(64, 64).T)
    '''
