# coding=utf-8
import os
import random
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet34
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import pandas as pd
from multiprocessing import Pool


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.shape = 0

    def forward(self, x):
        self.shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, self.shape)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.trained_model = resnet34(pretrained=True)  # .to(device)
        self.model1 = nn.Sequential(*list(self.trained_model.children())[:-1],  # 测试一下输出维度[b, 512, 1, 1]
                                    Flatten(),
                                    nn.Linear(512, 128),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(128, 32),
                                    )
        # self.trained_model2 = vgg16(pretrained=True)  # .to(device)
        self.model2 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    # 测试一下输出维度[b, 512, 1, 1]
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.Dropout(p=0.4),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.4),
                                    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.4),
                                    nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.AdaptiveAvgPool2d(output_size=(7, 7)),
                                    Flatten(),
                                    nn.Linear(12544, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 32)
                                    )
        self.model3 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, input1, input2):
        output11 = self.model1(input1)
        output12 = self.model2(input1)
        output1 = torch.cat([output11, output12], dim=1)
        output21 = self.model1(input2)
        output22 = self.model2(input2)
        output2 = torch.cat([output21, output22], dim=1)
        return output1, output2


# 自定义ContrastiveLoss
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    同类为0， 不同类为1
    """

    def __init__(self, margin=3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255  # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img


class ReaderSiameseDatas(Dataset):
    """
    一类图片放到一个文件夹里，文件夹名对应图片类别
    mode='train'时，生成训练集；mode='test'时生成测试集
    """

    def __init__(self, picture_root='datas', mode="train", resize=225, sample_num=4000, ratio=0.7, random_seed=10):
        super(ReaderSiameseDatas, self).__init__()
        self.random_seed = random_seed
        # 生成的样本数量
        self.sample_num = sample_num
        self.picture_root = picture_root
        # 训练集与验证集之比
        self.ratio = ratio
        # 图片新尺寸大小
        self.resize = resize
        self.mode = mode
        # self.images中存储图片对(路径)，用于训练
        self.images = []
        # self.labels存储self.images的图片对标签，只含0、1
        self.labels = []
        # 用于测试
        self.test_images = []
        self.test_labels = []
        # 用于测试的基准库
        self.base_images = []
        self.base_labels = []
        self.category_num = len(os.listdir(self.picture_root))
        # 多个列表，每个列表中为某一类的图片地址
        self.temp_images = []
        self.temp_labels = []
        self.make_tag()
        self.tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((self.resize, self.resize)),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.RandomRotation(45),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.435, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.tf2 = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.435, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.random_seed = random_seed

    def make_tag(self):

        for file in os.listdir(os.path.join(self.picture_root, self.mode)):
            self.temp_images.append([])
            self.temp_labels.append([])
            for img in os.listdir(os.path.join(self.picture_root, self.mode, file)):
                img = os.path.join(self.picture_root, self.mode, file, img)
                self.temp_images[int(file)].append(img)
                self.temp_labels[int(file)].append(int(file))

        if self.mode == 'test':
            self.test_images = self.temp_images[0] + self.temp_images[1]
            self.test_labels = self.temp_labels[0] + self.temp_labels[1]

        if self.mode == 'train':
            self.base_images = self.temp_images[0] + self.temp_images[1]
            self.base_labels = self.temp_labels[0] + self.temp_labels[1]

        # 构造标签对
        # 增加同类图片对和标签
        for i in range(3):
            for imgg in self.temp_images[0]:
                self.images.append([imgg, random.choice(self.temp_images[0])])
                self.labels.append(int(0))
        # for i in range(int(self.sample_num/300)):
        #     self.images.append(random.sample(self.temp_images[1], 2))
        #     self.labels.append(int(0))
        # 增加非同类图片对和标签
        for i in range(int(self.sample_num)):
            self.images.append([random.choice(self.temp_images[0]), random.choice(self.temp_images[1])])
            self.labels.append(int(1))

        # 打散
        random.seed(self.random_seed)
        random.shuffle(self.images)
        random.seed(self.random_seed)
        random.shuffle(self.labels)

    def get_base_tensors(self):
        """
        读取多维列表中的每一个图片路径， 然后转化为tensor
        :return:
        """
        base_tensors = []
        for lst in self.temp_images:
            temp_tensors = []
            for img in lst:
                img = self.tf(img)
                temp_tensors.append(img)
            base_tensors.append(temp_tensors)
        return base_tensors

    def get_test_tensors(self):
        """
        读取测试列表中的图片路径，转化为tensor
        :return:
        """
        test_tensors = []
        for img in self.test_images:
            img = self.tf2(img)
            test_tensors.append(img)
        return test_tensors

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        if self.mode == "train":
            img1, img2, label = self.images[idx][0], self.images[idx][1], self.labels[idx]
            img1 = self.tf(img1)
            img2 = self.tf(img2)
            label = torch.tensor(int(label))
            return img1, img2, label
        if self.mode == "test":
            img, label = self.test_images[idx], self.test_labels[idx]
            img = self.tf(img)
            label = torch.tensor(int(label))
            return img, label


batchsz = 4
lr = 4e-5
epochs = 100


def get_vectors(model, tensors):
    """
    模型训练好后，用于生成每个base图片的特征向量
    :param model:
    :param images: 内含多个列表，子列表中存储图片的初始tensor
    :return: 输入每个图片经过模型后的向量
    """
    vectors_list = []
    model.eval()
    # model = model.cup()
    for i in range(len(tensors)):
        temp_vectors = []
        with torch.no_grad():
            for tensor in tensors[i]:
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                    tensor = tensor.to(device)
                # 模型的输入是四维，因此需要增加一个维度
                tensor = tensor.unsqueeze(0)
                vector, _ = model(tensor, tensor)
                vector = vector.squeeze(0)
                vector = vector.cpu()
                temp_vectors.append(vector)
        vectors_list.append(temp_vectors)
    return vectors_list


def get_predict_label(base_vectors, vector, length=10):
    """

    :param base_vectors: 内含多个子列表，子列表中存储模型输出向量
    :param vector:某张图片经过模型后得到的一维向量
    :return:输出类别（只有一个整数，譬如：0）
    """
    distances = []
    for i in range(len(base_vectors)):
        temp_distances = []
        for vec in base_vectors[i]:
            temp_distances.append(F.pairwise_distance(vec.unsqueeze(0), vector.unsqueeze(0), keepdim=True))
        temp_distances = sorted(temp_distances)
        # print("temp_distances:", temp_distances)
        distances.append(temp_distances)
    label = 0
    print("基准距离0", distances[0][0:length])
    print("基准距离1", distances[1][0:5])
    dis = sum(distances[0][0:length]) / length
    for i in range(1, len(distances)):
        if dis >= distances[i][0]:
            label = int(i)
            dis = distances[i][0]
            continue
    return label


def main():
    train_db = ReaderSiameseDatas(mode='train')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=False, num_workers=0)
    test_db = ReaderSiameseDatas(mode="test")
    base_tensors = train_db.get_base_tensors()
    test_tensors = test_db.get_test_tensors()

    model = SiameseNetwork()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = ContrastiveLoss()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('cuda:', use_gpu)
        device = torch.device('cuda:0')
        model = model.to(device)
        criteon = criteon.to(device)

    best_f1 = 0
    best_epoch = 0
    for epoch in tqdm(range(epochs)):
        model.train()
        train_db = ReaderSiameseDatas(mode='train')
        train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=False, num_workers=0)
        for step, (img1, img2, label) in enumerate(train_loader):
            if use_gpu:
                device = torch.device('cuda:0')
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            logit1, logit2 = model(img1, img2)
            loss = criteon(logit1, logit2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:

            # 获取预测标签
            model.eval()
            base_vectors = get_vectors(model, base_tensors)
            predict_label = []
            for img in test_tensors:
                img = img.unsqueeze(0)
                if use_gpu:
                    device = torch.device('cuda:0')
                    img = img.to(device)
                a, _ = model(img, img)
                img, _ = model(img, img)
                # if torch.all(torch.eq(img, a)):
                #     print('模型输出结果唯一')
                # else:
                #     print('模型结果不唯一')
                img = img.squeeze(0)
                img = img.to('cpu')
                predict_label.append(get_predict_label(base_vectors, img))
            f1 = f1_score(list(test_db.test_labels), list(predict_label), average='macro')
            if f1 >= 0.7:
                optimizer = optim.Adam(model.parameters(), lr=1e-5)
            else:
                optimizer = optim.Adam(model.parameters(), lr=4e-5)
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), 'best.mdl')
                report = classification_report(list(test_db.test_labels), list(predict_label), output_dict=True)
                df = pd.DataFrame(report).transpose()
                df.to_csv("repost.csv", index=True)
            # print("测试集")
            # print('损失：', loss)
            # print("预测值：", predict_label)
            # print("真实值：", test_db.test_labels)
            for i in range(len(predict_label)):
                if predict_label[i] != test_db.test_labels[i]:
                    print('真实值：', test_db.test_labels[i], '预测值：', predict_label[i], '图片名称：', test_db.test_images[i])
            target_names = ['正常', '灰黑']
            print(classification_report(list(test_db.test_labels), list(predict_label), target_names=target_names))

    # 加载参数
    model.load_state_dict(torch.load('best.mdl'))
    test_f1 = f1_score(list(test_db.test_labels), list(predict_label), average='macro')
    print("test_macro-f1:", test_f1)
    print("最佳模型报告：\n", report)
    print("最佳epoch：", best_epoch)


if __name__ == "__main__":
    main()
