import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.shape = 0

    def forward(self, x):
        self.shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, self.shape)


class Ssvdd(nn.Module):
    def __init__(self):
        super(Ssvdd, self).__init__()

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
                                    nn.Linear(12544, 2000),
                                    nn.ReLU(),
                                    nn.Linear(2000, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 32)
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


def preprocess_img(original_img):
    """
    :param original_img: RGB（PIL）
    :return:torch.tensor格式，shape=[b,c,h,w]
    """
    resize = 225
    preprocessing = transforms.Compose([
        # lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.435, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    processed_img = preprocessing(original_img)
    processed_img = processed_img.unsqueeze(0)
    return processed_img


def predict_label(base_vectors, vector, length=10):
    """

    :param base_vectors: 内含多个子列表，子列表中存储模型输出向量
    :param vector:某张图片经过模型后得到的一维向量
    :return:输出类别（只有一个整数，譬如：0）
    """
    # vector = torch.from_numpy(vector)
    distances = []
    for i in range(1):
        temp_distances = []

        for vec in base_vectors[i]:
            temp_distances.append(F.pairwise_distance(vec.unsqueeze(0), vector.unsqueeze(0), keepdim=True))
        temp_distances = sorted(temp_distances)
        # print("temp_distances:", temp_distances)
        distances.append(temp_distances)
    label = 0
    # print("基准距离0", distances[0][0:length])
    dis = sum(distances[0][0:length])/length

    if dis > 0.35:
        label = 1
    return label, dis


def infer(arr, input_model):
    base_vectors = np.load('base_vectors.npy', allow_pickle=True)
    input_img = Image.fromarray(arr).convert("RGB")
    input_img = preprocess_img(input_img)

    img = input_img
    img, _ = input_model(img, img)
    img = img.squeeze(0)
    img = img.cpu()
    lab, dis = predict_label(base_vectors, img)
    print(lab, dis)
    return lab


class DeploySsvdd(object):
    def __init__(self):
        device = torch.device("cpu")  # 限定使用cpu
        self.model = Ssvdd()
        path_tmp = os.path.join(os.path.dirname(__file__), "best.mdl")
        self.model.load_state_dict(torch.load(path_tmp, map_location=device))
        self.model.eval()
        self.model.to(device=device)

    def run(self, arr: np.ndarray):
        label = infer(arr, self.model)
        return label


if __name__ == "__main__":
    # 加载图片为RGB格式
    picture_root = r'p10001131.png'
    img = Image.open(picture_root).convert('RGB')

    ts1 = time.time()
    obj = DeploySsvdd()
    ts2 = time.time()
    print("ts1: {}".format(ts2 - ts1))

    for _ in range(10):
        pred = obj.run(np.array(img))
        print(pred)
    ts = time.time() - ts2
    print("ts1: {}".format(ts / 10))
