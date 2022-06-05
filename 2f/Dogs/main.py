import tarfile
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, homogeneity_score, completeness_score, accuracy_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import trange
import random


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


def separate_data(data, labels):
    indexes_dogs = np.where(labels.values.ravel() == 1)[0]
    indexes_cats = np.where(labels.values.ravel() == 0)[0]
    dogs = data.iloc[indexes_dogs]
    label_dogs = pd.DataFrame(np.ones(np.shape(dogs)[0]))
    cats = data.iloc[indexes_cats]
    label_cats = pd.DataFrame(np.zeros(np.shape(cats)[0]))

    return dogs, label_dogs, cats, label_cats

def plot_pic(pics, i):
    for pic in pics:

        pic1 = np.array(pic)
        plt.title(f'Cluster number = '+str(i))
        plt.imshow(pic1.reshape(64, 64).T)
        plt.show()

def get_pics(y_pred, data, cluster):
    indexes = np.where(y_pred == cluster)[0]
    pics = data.iloc[indexes]

    return np.array(pics), indexes

def get_rot_data(data):
    num_pics = int(len(data) / 2)
    pics = random.sample(range(0, len(data)), len(data))
    for ind in pics:
        pic = np.array(data.iloc[ind]).reshape(64, 64)
        data.iloc[ind] = np.rot90(pic).flatten()
    data = pd.DataFrame(data)
    return data, pics



def main(data):
    data=pd.DataFrame(np.array(data).T)


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
    data_pca = pd.DataFrame(pca.fit_transform(data))

    #x_tr, x_te, y_tr, y_te = train_test_split(data, label, test_size=0.2, random_state=8)
    #TODO träna på trainoch kör på test

    '''
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(data)

    y_pred = kmeans.predict(data)
    '''
    num_runs = 1
    num_cluster_list = range(15, 16)
    silhouette = np.zeros([num_runs, len(num_cluster_list)])
    calinski = np.zeros([num_runs, len(num_cluster_list)])
    davies = np.zeros([num_runs, len(num_cluster_list)])
    for i in trange(num_runs):
        for j, num_clusters in enumerate(num_cluster_list):
            gmm = GaussianMixture(n_components=num_clusters)
            y_pred = gmm.fit_predict(data_pca)

            # y_pred = fix_pred(y_pred, label_dogs, num_clusters)

            # print(f'accuracy = {accuracy_score(label_cats, y_pred)}, silhuette score = {silhouette_score(cats, y_pred)}')
            # print(homogeneity_score(np.array(label_dogs).flatten(), y_pred))
            silhouette[i, j] = silhouette_score(data_pca, y_pred)
            calinski[i, j] = calinski_harabasz_score(data_pca, y_pred)
            davies[i, j] = davies_bouldin_score(data_pca, y_pred)

            pics = [get_pics(y_pred, data, i)[0] for i in range(num_clusters)]
            indexes = [get_pics(y_pred, data, i)[1] for i in range(num_clusters)]

            index_min = 0
            print(indexes)
            # run_1b(pd.DataFrame(indexes[index_min]))
            plot_list = np.zeros(4096)
            for index in indexes:
                plot_list[index]=y_pred[index]
            plt.figure()
            plt.imshow(plot_list.reshape(64,64).T)
            #plot_pic(pics[index_min], i=index_min)
            plt.show()



    print(silhouette)
    silhouette_sc = np.average(silhouette, axis=0)
    davies_sc = np.average(davies, axis=0)
    calinski_sc = np.average(calinski, axis=0)
    plt.figure()
    plt.title('Siluette')
    plt.ylabel('score')
    plt.xlabel('#clusters')
    plt.plot(num_cluster_list, silhouette_sc)
    plt.figure()
    plt.title('Davies-Bouldin')
    plt.ylabel('score')
    plt.xlabel('#clusters')
    plt.plot(num_cluster_list, davies_sc)
    plt.figure()
    plt.title('Calinski-Harabasz')
    plt.ylabel('score')
    plt.xlabel('#clusters')
    plt.plot(num_cluster_list, calinski_sc)
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
    main(dogs)
