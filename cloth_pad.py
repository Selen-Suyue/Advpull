import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from attack_train import InpaintingNetwork


haha=torch.load('generator.pt')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ])

image_paths1 = []

for root, dirs, files in os.walk('SYSU\\cam1\\0001'):
    for file in files:
        image_paths1.append(os.path.join(root, file))

image1 = Image.open(image_paths1[0])
image11 = preprocess(image1)
image11 = image11.unsqueeze(0)
image11=image11.cuda()
patch = haha(image11)
a = patch.squeeze(0)


tensor = a.cpu()

img = tensor.data.numpy()
img=img*200

img_rgb = np.zeros((30, 30, 3))


img_rgb[:, :, 0] = img[0, :, :]
img_rgb[:, :, 1] = img[1, :, :]
img_rgb[:, :, 2] = img[2, :, :]


img_rgb[img_rgb > 255] = 255

cv2.imwrite('output_image.jpg', img_rgb)




