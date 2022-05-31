import tarfile
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, homogeneity_score, completeness_score, accuracy_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import trange
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel


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


def plot_feat(feat_list, method, data, label):
    plt.figure()
    pic1 = np.array(data.iloc[0])
    plt.title(f'label = {get_label(np.array(label.iloc[0]))}, '+method)
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


def fix_pred(pred, true_val, num_clusters):
    pred_new=np.zeros(len(pred))
    for i in range(num_clusters):
        y = sum(true_val.iloc[np.where(pred == i)[0]].values.ravel())/len(true_val.iloc[np.where(pred == i)[0]].values.ravel())
        if y > 0.5:
            pred_new[np.where(pred == i)[0]] = 1
        else:
            pred_new[np.where(pred == i)[0]] = 0
    return pred_new


def separate_data(data, labels):
    indexes_dogs = np.where(labels.values.ravel() == 1)[0]
    indexes_cats = np.where(labels.values.ravel() == 0)[0]
    dogs = data.iloc[indexes_dogs]
    label_dogs = pd.DataFrame(np.ones(np.shape(dogs)[0]))
    cats = data.iloc[indexes_cats]
    label_cats = pd.DataFrame(np.zeros(np.shape(cats)[0]))

    return dogs, label_dogs, cats, label_cats

def run_1a(data, label):
    np_data = np.array(data)
    np_labels = np.array(label).flatten()

    wrong_list = []
    mean_score = []
    std = []
    tpr = []
    tnr = []
    x_tr, x_te, y_tr, y_te = train_test_split(data, label, test_size=0.1, random_state=8)

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
                #tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
                confusion_mat = confusion_matrix(y_test, prediction)
                fp = confusion_mat.sum(axis=0) - np.diag(confusion_mat)
                fn = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
                tp = np.diag(confusion_mat)
                tn = confusion_mat.sum() - (fp + fn + tp)
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


def run_1b(data, label):

    np_data = np.array(data)
    np_labels = np.array(label).flatten()

    num_runs = 100
    threshold = 0.8

    method=['Variance', 'SelectKbest', 'Linear svc']

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
    plt.bar(range(np.shape(data)[1]), count_feat[0, :])
    plt.figure()
    plt.title('Select K best, chi2 score')
    plt.bar(range(np.shape(data)[1]), count_feat[1, :])
    plt.figure()
    plt.title('Linear svc')
    plt.bar(range(np.shape(data)[1]), count_feat[2, :])

    count_feat[count_feat < num_runs*threshold] = 0

    for i in range(len(count_feat[:, 0])):
        feat_list = np.nonzero(count_feat[i, :])
        plot_feat(feat_list[0], method[i], data, label)

    plt.show()


if __name__ == '__main__':
    try:
        data = pd.read_csv("/Users/antonrosenberg/Documents/GitHub/BigDTenta/CATSnDOGS.csv") / 255
        label = pd.read_csv("/Users/antonrosenberg/Documents/GitHub/BigDTenta/Labels.csv")
    except:
        data = pd.read_csv("C:/Users/anton\OneDrive\Dokument\GitHub\BigDTenta/CATSnDOGS.csv") / 255
        label = pd.read_csv("C:/Users/anton\OneDrive\Dokument\GitHub\BigDTenta/Labels.csv")

    np_data = np.array(data)
    np_labels = np.array(label).flatten()

    dogs, label_dogs, cats, label_cats = separate_data(data, label)

    pca = PCA()
    #pca.fit(cats)
    pca.fit(dogs)
    cut_off = 0.03
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2)
    plt.axhline(cut_off, color="red")
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained Ratio')


    num_pca = len(pca.explained_variance_ratio_[pca.explained_variance_ratio_ > cut_off])
    print(f'num_pca choosen = {num_pca}')
    pca = PCA(n_components=num_pca)
    cats_pca = pd.DataFrame(pca.fit_transform(cats))
    dogs_pca = pd.DataFrame(pca.fit_transform(dogs))

    #x_tr, x_te, y_tr, y_te = train_test_split(data, label, test_size=0.2, random_state=8)
    #TODO träna på trainoch kör på test

    #num_clusters = 5 # Cats
    num_clusters = 2 # Dogs

    gmm = GaussianMixture(n_components=num_clusters)
    #y_pred = pd.DataFrame(gmm.fit_predict(cats_pca))
    y_pred = pd.DataFrame(gmm.fit_predict(dogs_pca))
    print(type(y_pred))

    #run_1a(cats_pca, y_pred)
    run_1a(dogs_pca, y_pred)

    run_1b(dogs, y_pred)





