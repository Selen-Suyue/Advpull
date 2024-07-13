import os
import torch
from torchvision import models
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from lv import calculate_lv
import math

patch_size=30
model = models.vgg16(pretrained=True)

model.cuda()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
normal=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def merge_images(A, B):
    B = F.pad(B, [97, 97, 97, 97], mode='constant')
    return A+B

def compute_feature_distance(image1, image2):
  # 图像预处理
  image1 = normal(image1)
  image2 = preprocess(image2)


  image1 = image1.cuda()
  image2 = image2.cuda()


  features1 = model.features(image1)
  features2 = model.features(image2)


  distance = torch.nn.functional.mse_loss(features1, features2)

  return distance


image_paths1 = []
for root, dirs, files in os.walk('SYSU\\cam1\\0001'):
    for file in files:
      image_paths1.append(os.path.join(root, file))

image_paths2 = []
for root, dirs, files in os.walk('SYSU\\cam6\\0001'):
    for file in files:
      image_paths2.append(os.path.join(root, file))

class InpaintingNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 32, (3, 3), stride=1, padding='same', bias=False)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=1, padding='same', bias=False)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 30 * 30*30)
        self.conv3 = nn.Conv2d(30, 3, (3, 3), stride=1, padding='same', bias=False)

    def forward(self, x):

        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 30, 30, 30)
        x = self.conv3(x)
        x = nn.ReLU()(x)

        return x



# generator = InpaintingNetwork()
# generator.cuda()
# optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
#
#
# for epoch in range(9):
#     tance = 0
#     for i in range(len(image_paths1)):
#         image1 = Image.open(image_paths1[i])
#         for j in range(len(image_paths2)):
#             image2 = Image.open(image_paths2[j])
#             image11= preprocess(image1)
#             image11=image11.unsqueeze(0)
#             image11=image11.cuda()
#             patch=generator(image11)
#             a=patch.squeeze(0)
#             a=normal(a)
#             image11 = image11.squeeze(0)
#             loss0 = compute_feature_distance(image11, image2)
#             image11=merge_images(image11,a)
#             loss1 = compute_feature_distance(image11, image2)
#             loss2 = calculate_lv(image11)
#             loss3 = math.exp(loss0)-math.exp(loss1)
#             tance=loss1+loss2+loss3
#             optimizer.zero_grad()
#             loss = tance
#             loss.backward()
#             optimizer.step()
#
# torch.save(generator,'generator.pt')








