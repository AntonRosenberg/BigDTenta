import tarfile
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from tqdm import trange
from sklearn.model_selection import GridSearchCV
import random

def histogram(wrong_list):
    histo = np.zeros(198)
    for arr in wrong_list:
        for element in arr:
            histo[element]+=1
    return histo

def get_label(label):
    if label==1:
        return 'Dog'
    return 'Cat'

def get_scores(array, start):
    sc=[]
    for arr in array:
        sc.append(arr[start])
    sc = sum(sc)/len(sc)
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
        #plt.title(f'label = {label[label.index==errors]}')
        plt.imshow(pic)
        plt.show()

def add_noise(data, noise_level, num_pics):
    pics = random.sample(range(0, len(data)), num_pics)
    for ind in pics:
        for i in range(len(data.iloc[0])):
            #noise = np.random.randint(noise_level) / 255
            noise = 1
            if np.random.rand() < 0.5:
                data.iloc[ind][i] -= noise
                if data.iloc[ind][i] < 0:
                    data.iloc[ind][i] = 0
            else:
                data.iloc[ind][i] += noise
                if data.iloc[ind][i] > 1:
                    data.iloc[ind][i] = 1
    return data, pics

def get_16X16(data, box_num):
    box_ind = np.linspace(box_num*255, 255*(box_num+1), 256)
    box_ind = [int(val) for val in box_ind]
    boxes = [np.zeros(256) for _ in range(len(data))]

    indexes = range(box_num*256, 256*(box_num+1), 16)
    indexes_boxes = range(0, 256, 16)


    if box_num<4:
        index_row = range(box_num * 16, 16 * (1 + box_num))
        index_column = range(0, 16)
    elif box_num<8:
        index_row = range((box_num-4) * 16, 16 * (1 + (box_num-4)))
        index_column = range(16, 32)
    elif box_num<12:
        index_row = range((box_num-8) * 16, 16 * (1 + (box_num-8)))
        index_column = range(32, 48)
    else:
        index_row = range((box_num-12) * 16, 16 * (1 + (box_num-12)))
        index_column = range(48, 64)
    ind_add = 49

    for i, pic in enumerate(data):
        pic= pic.reshape(64,64)
        for j, row in enumerate(index_row):
            for k, col in enumerate(index_column):
                boxes[i][k+16*j] = pic[row][col]

    return pd.DataFrame(boxes)

def get_hyper(model, X, y, params):

    results = pd.DataFrame(columns=['model', 'params'])



    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X, y.values.ravel())
    row = {'model': model, 'params': clf.best_params_}
    print(clf.best_params_['C'])


    return clf.best_params_['C']

def main(num_runs, box_num):
    try:
        data = pd.read_csv("/Users/antonrosenberg/Documents/GitHub/BigDTenta/CATSnDOGS.csv") / 255
        label = pd.read_csv("/Users/antonrosenberg/Documents/GitHub/BigDTenta/Labels.csv")
    except:
        data = pd.read_csv("C:/Users/anton\OneDrive\Dokument\GitHub\BigDTenta/CATSnDOGS.csv") / 255
        label = pd.read_csv("C:/Users/anton\OneDrive\Dokument\GitHub\BigDTenta/Labels.csv")

    np_data = np.array(data)
    np_labels = np.array(label).flatten()

    data = get_16X16(np_data, box_num)

    wrong_list = []
    mean_score = []
    std = []
    tpr = []
    tnr = []
    x_tr, x_te, y_tr, y_te = train_test_split(data, label, test_size=0.1, random_state=8)

    k = 5
    models = [SVC(), RandomForestClassifier(), LogisticRegression(solver='liblinear', max_iter=1000, penalty='l2')]
    params_SVC = get_hyper(models[0], x_tr, y_tr, params = [{'C': np.logspace(-4,4,30)}])
    params_LogReg = get_hyper(models[2], x_tr, y_tr, params = [{'C': np.logspace(-4,4,30)}])

    for j in trange(num_runs):
        kf = StratifiedKFold(n_splits=k, shuffle=True)
        prediction = []
        models = [SVC(C=params_SVC), RandomForestClassifier(),
                  LogisticRegression(C=params_LogReg, max_iter=1000, penalty='l2')]
        score = np.zeros([len(models), k])
        fold_ind = 0
        for train_index, test_index in kf.split(x_tr, y_tr):
            X_train, X_test = x_tr.iloc[train_index], x_tr.iloc[test_index]
            y_train, y_test = y_tr.iloc[train_index], y_tr.iloc[test_index]
            for ind, model in enumerate(models):
                model.fit(X_train, y_train.values.ravel())
                prediction = model.predict(X_test)
                score[ind, fold_ind]=(model.score(x_te, y_te))
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

    file = open(f'Score for , box num = {box_num}', 'w')
    file.write( f'SVM: accuracy = {mean_score1}, std = {std1}, tpr = {tpr1}, tnr = {tnr1} \n '
          f'RandomForest: accuracy = {mean_score2}, std = {std2}, tpr = {tpr2}, tnr = {tnr2} \n'
          f'LogisticRegression: accuracy = {mean_score3}, std = {std3}, tpr = {tpr3}, tnr = {tnr3}')

    file.close()

    #np.savetxt(f'Score for , noise = {noise_level}, # noisy pics = {num_pics}', f'SVM: accuracy = {mean_score1}, std = {std1}, tpr = {tpr1}, tnr = {tnr1} \n '
    #     f'RandomForest: accuracy = {mean_score2}, std = {std2}, tpr = {tpr2}, tnr = {tnr2} \n'
    #      f'LogisticRegression: accuracy = {mean_score3}, std = {std3}, tpr = {tpr3}, tnr = {tnr3}')

    plt.figure()
    plt.title(f'SVM, box num = {box_num}')
    plt.xlabel('Pic #')
    plt.ylabel('Wrongly classified count')
    plt.bar(range(len(histo1)), histo1)
    plt.savefig(f'SVM, box num = {box_num}')
    plt.figure()
    plt.title(f'RandomForest, box num = {box_num}')
    plt.xlabel('Pic #')
    plt.ylabel('Wrongly classified count')
    plt.bar(range(len(histo1)), histo2)
    plt.savefig(f'RandomForest, box num = {box_num}')
    plt.figure()
    plt.title(f'LogisticRegression, box num = {box_num}')
    plt.xlabel('Pic #')
    plt.ylabel('Wrongly classified count')
    plt.bar(range(len(histo1)), histo3)
    plt.savefig(f'LogisticRegression, box num = {box_num}')
    plt.close('all')


    '''
    ones = np.size(label.index[label.index == 1])
    zeroes = np.size(label.index[label.index == 0])
    
    scores = [cross_val_score(model, data, label.values.ravel(), cv=5, scoring='accuracy') for model in models]
    
    print(scores)

    wrong_list = [check_predictions(model.fit(x_train, y_train).predict(x_test), y_test) for model in models]
    print(wrong_list)

    plot_err(wrong_list[0], data)
    '''



if __name__ == '__main__':
    box_num_list=range(16)
    num_runs = 100
    for box_num in box_num_list:
        main(num_runs, box_num)





