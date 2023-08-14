from sklearn import svm
from sklearn.metrics import classification_report
import joblib
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from siamese_network import ReaderSiameseDatas
from siamese_network import get_predict_label, get_vectors
from siamese_network import SiameseNetwork
from siamese_network import Flatten

batchsz = 32


def get_vectors_and_labels():
    model = SiameseNetwork()
    model.load_state_dict(torch.load('best.mdl'))

    train_db = ReaderSiameseDatas(mode='train')
    base_tensors = train_db.get_base_tensors()

    base_labels = train_db.temp_labels[0] + train_db.temp_labels[1]
    try:
        vectors = np.load('vectors.npy')
    except FileNotFoundError:
        device = torch.device('cuda:0')
        model.to(device)
        base_vectors = get_vectors(model, base_tensors)
        base_vectors = base_vectors[0] + base_vectors[1]
        vectors = []
        for i in base_vectors:
            device1 = torch.device('cpu')
            i = i.to(device1)
            vectors.append(i.numpy())
        np.save('vectors.npy', vectors)

    return vectors, base_labels


def train_and_save_svm():
    vec, lab = get_vectors_and_labels()
    clf = svm.SVC(kernel='rbf', class_weight={1: 1})
    clf.fit(vec, lab)
    joblib.dump(clf, "svm_model.m")

    return clf


def get_svm_model():
    try:
        clf = joblib.load("svm_model.m")
    except FileNotFoundError:
        clf = train_and_save_svm()

    return clf


def svm_predict(clf, vector):
    return clf.predict(vector)


def test_svm():
    device = torch.device('cuda:0')
    model = SiameseNetwork()
    model.load_state_dict(torch.load('best.mdl'))
    model.to(device)
    test_db = ReaderSiameseDatas(mode="test")
    test_tensors = test_db.get_test_tensors()
    true_label = test_db.test_labels
    pred = []
    score = []
    for i in test_tensors:
        i = i.unsqueeze(0)
        i = i.to(device)
        vectors, _ = model(i, i)
        # vectors = vectors.squeeze(0)
        device1 = torch.device('cpu')
        vectors = vectors.to(device1)
        i = svm_predict(get_svm_model(), vectors.detach().numpy())
        score.append(i)
        if i > 0.5:
            i = 1
        else:
            i = 0
        pred.append(i)

    print(score)
    print(pred)
    print(true_label)

    print(classification_report(true_label, pred))


def get_center_point(base_vector):
    vec = torch.zeros(1, 64)
    for i in range(len(base_vector)):
        vec = vec + base_vector[0][i]
    vec = vec / len(base_vector)
    print(vec)
    return vec


# def predict_label(base_vectors, vector, length=10):
#     """
#
#     :param base_vectors: 内含多个子列表，子列表中存储模型输出向量
#     :param vector:某张图片经过模型后得到的一维向量
#     :return:输出类别（只有一个整数，譬如：0）
#     """
#     # vector = torch.from_numpy(vector)
#     distances = []
#     for i in range(1):
#         temp_distances = []
#
#         for vec in base_vectors[i]:
#             temp_distances.append(F.pairwise_distance(vec.unsqueeze(0), vector.unsqueeze(0), keepdim=True))
#         temp_distances = sorted(temp_distances)
#         # print("temp_distances:", temp_distances)
#         distances.append(temp_distances)
#     label = 0
#     # print("基准距离0", distances[0][0:length])
#     dis = sum(distances[0][0:length])/length
#
#     if dis > 0.35:
#         label = 1
#     return label, dis


def predict_label(vec, vector):
    """

    :param vec: 内含多个子列表，子列表中存储模型输出向量
    :param vector:某张图片经过模型后得到的一维向量
    :return:输出类别（只有一个整数，譬如：0）
    """
    label = 0
    dis = F.pairwise_distance(vec.unsqueeze(0), vector.unsqueeze(0), keepdim=True)

    if dis > 0.5:
        label = 1
    return label, dis


def test_svdd():
    use_gpu = torch.cuda.is_available()
    train_db = ReaderSiameseDatas(mode='train')
    test_db = ReaderSiameseDatas(mode="test")
    base_tensors = train_db.get_base_tensors()
    test_tensors = test_db.get_test_tensors()
    true_label = test_db.test_labels

    device = torch.device('cpu')
    if use_gpu:
        device = torch.device('cuda:0')
    model = SiameseNetwork()
    model.load_state_dict(torch.load('best.mdl'))
    model.to(device)

    model.eval()
    try:
        base_vectors = np.load('base_vectors.npy', allow_pickle=True)
        print('fdsfd')
    except:
        print('kkkkk')
        base_vectors = get_vectors(model, base_tensors)
        # base_vectors = base_vectors.cpu()
        np.save('base_vectors.npy', base_vectors)

    pred = []
    score = []
    vec = get_center_point(base_vectors)
    for img in tqdm(test_tensors):
        img = img.unsqueeze(0)
        img = img.to(device)
        img, _ = model(img, img)
        img = img.squeeze(0)
        img = img.cpu()
        vec = torch.tensor([[0.2082, -0.1405, -0.2509, -0.1142, 0.2315, -0.6253, -0.1543, 0.0305,
                             0.0032, 0.1574, -0.3434, -0.1140, 0.0548, -0.3768, -0.1477, 0.3405,
                             0.1099, -0.3561, -0.1902, -0.4951, -0.4137, 0.1315, -0.0788, -0.5875,
                             0.3403, -0.0112, -0.0164, 0.1749, -0.0888, -0.1353, 0.5193, -0.0017,
                             -0.0275, 0.0731, -0.0674, 0.0039, 0.0798, 0.0539, 0.0331, 0.0758,
                             0.0764, 0.0035, -0.0035, -0.0958, 0.0944, -0.0896, -0.0294, 0.0362,
                             -0.1040, -0.0015, 0.0238, -0.0083, -0.0138, -0.0130, -0.0377, -0.0088,
                             -0.0803, 0.0898, 0.0359, 0.0635, 0.0270, -0.1072, 0.0041, -0.0888]])
        lab, dis = predict_label(vec, img)
        pred.append(lab)
        score.append(float(dis))

    print(score)
    print(classification_report(true_label, pred))


if __name__ == '__main__':
    test_svdd()
    # test_svm()
