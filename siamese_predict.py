# coding=utf-8
import os

import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision import transforms
from PIL import Image
import numpy as np


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
                                    nn.Linear(512, 32)
                                    )

    def forward(self, input1, input2):
        output1 = self.model1(input1)
        output2 = self.model1(input2)
        return output1, output2


# 加载模型
model = SiameseNetwork()
model.load_state_dict(torch.load('res34.mdl', map_location='cuda:0'))
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')
    model = model.to(device)


def get_vec(img_root, resize=256):
    """
    获取图片经过模型后的特征向量
    :param img_root: 图片路径
    :param resize: 改变尺寸大小
    :return: 特征向量
    """
    predict_tf = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.435, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    img = predict_tf(img_root)
    img = img.unsqueeze(0)
    if use_gpu:
        img = img.to(device)
    with torch.no_grad():
        vector, _ = model(img, img)
        vector = vector.squeeze(0)
    return vector


def get_and_save_base_vectors(root=r"D:\MyCodes\pythonProject\old_tender\datas\category"):

    base_vectors = []
    for i in range(len(os.listdir(root))):
        base_vectors.append([])
        for j in os.listdir(os.path.join(root, str(i))):
            vec = get_vec(os.path.join(root, str(i), j))
            vec = vec.cpu()
            base_vectors[i].append(vec)
    return base_vectors


def old_tender_predict(picture_root=r"D:\MyCodes\pythonProject\old_tender\datas\category\0\20.png"):
    """
    预测老嫩问题：0代表正常、1代表老、2代表嫩
    :param picture_root: 图片所在路径
    :return: 类别（格式为整数）
    """
    base_vectors = np.load('base_vectors.npy', allow_pickle=True)
    vector = get_vec(picture_root)
    vector = vector.cpu()
    distances = []
    for i in range(len(base_vectors)):
        temp_distances = []
        for vec in base_vectors[i]:
            temp_distances.append(torch.pairwise_distance(vec, vector, keepdim=True))
        temp_distances = sorted(temp_distances)
        # print("temp_distances:", temp_distances)
        distances.append(temp_distances)
    label = 0
    length = 1
    dis = sum(distances[0][0:length]) / length
    for i in range(1, len(distances)):
        if dis >= distances[i][0]:
            label = int(i)
            dis = distances[i][0]
            continue
    return int(label)


if __name__ == "__main__":
    a = old_tender_predict(r'D:\MyCodes\pythonProject\datas\all_seg_crop\400.png')
    print(a)

