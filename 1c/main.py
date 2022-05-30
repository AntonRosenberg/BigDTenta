import tarfile
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, homogeneity_score, completeness_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
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

def fix_pred(pred, true_val, num_clusters):
    pred_new=np.zeros(len(pred))
    for i in range(num_clusters):
        y = sum(true_val.iloc[np.where(pred == i)[0]].values.ravel())/len(true_val.iloc[np.where(pred == i)[0]].values.ravel())
        if y > 0.5:
            pred_new[np.where(pred == i)[0]] = 1
        else:
            pred_new[np.where(pred == i)[0]] = 0
    return pred_new


if __name__ == '__main__':
    try:
        data = pd.read_csv("/Users/antonrosenberg/Documents/GitHub/BigDTenta/CATSnDOGS.csv") / 255
        label = pd.read_csv("/Users/antonrosenberg/Documents/GitHub/BigDTenta/Labels.csv")
    except:
        data = pd.read_csv("C:/Users/anton\OneDrive\Dokument\GitHub\BigDTenta/CATSnDOGS.csv") / 255
        label = pd.read_csv("C:/Users/anton\OneDrive\Dokument\GitHub\BigDTenta/Labels.csv")

    np_data = np.array(data)
    np_labels = np.array(label).flatten()

    pca = PCA()
    pca.fit(data)
    cut_off = 0.03
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2)
    plt.axhline(cut_off, color="red")
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained Ratio')


    num_pca = len(pca.explained_variance_ratio_[pca.explained_variance_ratio_ > cut_off])
    print(num_pca)
    pca = PCA(n_components=num_pca)
    data = pd.DataFrame(pca.fit_transform(data))

    #x_tr, x_te, y_tr, y_te = train_test_split(data, label, test_size=0.2, random_state=8)
    #TODO träna på trainoch kör på test

    '''
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(data)

    y_pred = kmeans.predict(data)
    '''
    num_clusters = 2
    gmm = GaussianMixture(n_components=num_clusters)
    y_pred = gmm.fit_predict(data)

    y_pred = fix_pred(y_pred, label, num_clusters)

    print(accuracy_score(label, y_pred))
    print(homogeneity_score(np.array(label).flatten(), y_pred))

    data['labels'] = np_labels

    sns.pairplot(data, hue='labels',x_vars=[0, 1, 2, 3, 4], y_vars=[0, 1, 2, 3, 4])

    data['labels'] = y_pred

    sns.pairplot(data, hue='labels', x_vars=[0, 1, 2, 3, 4], y_vars=[0, 1, 2, 3, 4])

    plt.show()
    '''
    print(y_pred)
    print(homogeneity_score(np.array(label).flatten(), y_pred))
    print(completeness_score(np.array(label).flatten(), y_pred))

    cm = confusion_matrix(label.values.ravel(), y_pred)
    print(cm)
    '''

    '''
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

    for i in range(len(count_feat[:,0])):
        feat_list = np.nonzero(count_feat[i, :])
        plot_feat(feat_list[0], method[i])

    plt.show()
    '''

