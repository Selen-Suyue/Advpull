import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import numpy as np
import os
from PIL import Image
import random
import warnings


warnings.simplefilter("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SobelEdgeDetection(nn.Module):
    def __init__(self):
        super(SobelEdgeDetection, self).__init__()

        self.sobel_x = nn.Parameter(torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

        self.linear1 = nn.Linear(80*80, 128)
        self.linear2 = nn.Linear(128,32)
    def forward(self, x):

        x = TF.functional.rgb_to_grayscale(x)
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)

        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)

        edges = edges.view(edges.size(0), -1)

        x = self.linear1(edges)

        x = F.relu(x)

        x = self.linear2(x)

        x = torch.sigmoid(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, root_dir, cam_id, person_id, transform=None):
        self.root_dir = root_dir
        self.cam_id = cam_id
        self.person_id = person_id
        self.transform = transform
        self.file_list = self._load_file_list()
    def _load_file_list(self):
        file_list = []
        person_dir = os.path.join(self.root_dir, self.cam_id)
        for person_id in range(1, 2):
            person_folder = os.path.join(person_dir, str(person_id).zfill(4))
            if os.path.exists(person_folder):
                for file in os.listdir(person_folder):
                    if file.endswith('.jpg'):
                        file_list.append(os.path.join(person_folder, file))
        return file_list
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        image_path = self.file_list[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image.to(device)


transform = TF.Compose([
    TF.CenterCrop((80, 80)),
    TF.ToTensor(),
])

def extract_features_and_cluster(network, dataset):
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    features = []
    for images in dataloader:
        with torch.no_grad():
            images=images
            feature = network(images).cpu()
            features.append(feature)
    features = np.vstack(features)

    kmeans = KMeans(n_clusters=1, random_state=0).fit(features)
    return kmeans.cluster_centers_

def calculate_distance(cluster1, cluster2):
    return np.linalg.norm(cluster1 - cluster2)

def train_network(network, dataset_path, epochs=10):
    optimizer = torch.optim.Adam(network.parameters())
    for epoch in range(epochs):
        total_loss = 0
        for cam_id in ['cam1', 'cam6']:
            cam_path = os.path.join(dataset_path, cam_id)
            for i, person_id in enumerate(os.listdir(cam_path)):
                if i >= 100:
                    break
                person_id = int(person_id)

                dataset = CustomDataset(dataset_path, cam_id, person_id, transform=transform)

                random_indices = random.sample(range(len(dataset)), 3)
                selected_dataset = torch.utils.data.Subset(dataset, random_indices)

                cluster_centers = extract_features_and_cluster(network, selected_dataset)

                if cam_id == 'cam1':
                    cam1_centers = cluster_centers
                else:
                    distance = calculate_distance(cam1_centers, cluster_centers)
                    total_loss += distance

        loss = torch.tensor(total_loss, requires_grad=True).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    torch.save(network, 'edge_detection_model.pt')

def main():
    network = SobelEdgeDetection().to(device)
    train_network(network, 'SYSU')

if __name__=="__main__":
    main()

import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import numpy as np
import os
from PIL import Image
import random
import warnings

warnings.simplefilter("ignore")

torch.cuda.set_device(1)


# class HED(nn.Module):

#     def __init__(self, in_channels=1):
#         super(HED, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv9 = nn.Conv2d(512, 1, kernel_size=1)

#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(2, stride=2)
#         self.conv_adjust_channels4 = nn.Conv2d(
#             512, 1, kernel_size=1)  # Adjust channels
#         self.conv_adjust_channels3 = nn.Conv2d(256, 1, kernel_size=1)
#         self.conv_adjust_channels2 = nn.Conv2d(128, 1, kernel_size=1)
#         self.conv_adjust_channels1 = nn.Conv2d(64, 1, kernel_size=1)

#     def forward(self, x):
#         # Encoder
#         x1 = self.relu(self.conv1(x))
#         x1 = self.relu(self.conv2(x1))
#         x1 = self.maxpool(x1)  #1/2 1/2

#         x2 = self.relu(self.conv3(x1))
#         x2 = self.relu(self.conv4(x2))
#         x2 = self.maxpool(x2)  #1/4 1/4

#         x3 = self.relu(self.conv5(x2))
#         x3 = self.relu(self.conv6(x3))
#         x3 = self.maxpool(x3)  #1/8 1/8

#         x4 = self.relu(self.conv7(x3))
#         x4 = self.relu(self.conv8(x4))
#         x4 = self.maxpool(x4)  #1/16 1/16

#         # Decoder
#         x5 = self.conv9(x4)
#         x5 = self.relu(x5)
#         x4 = self.conv_adjust_channels4(x4)
#         x4 = self.relu(x4)
#         x3 = self.conv_adjust_channels3(x3)
#         x3 = self.relu(x3)
#         x2 = self.conv_adjust_channels2(x2)
#         x2 = self.relu(x2)
#         x1 = self.conv_adjust_channels1(x1)
#         x1 = self.relu(x1)

#         return x1, x2, x3, x4, x5


# class SEBlock(nn.Module):

#     def __init__(self, channels, reduction=1):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y).view(b, c, 1, 1)
#         return x * y


# class FeatureExtractor(nn.Module):

#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         self.hed = HED()
#         self.se_block1 = SEBlock(1)
#         self.se_block2 = SEBlock(1)
#         self.se_block3 = SEBlock(1)
#         self.se_block4 = SEBlock(1)
#         self.se_block5 = SEBlock(1)
#         self.conv_final = nn.Conv2d(1, 1, kernel_size=1)
#         self.linear = nn.Linear(40 * 40, 32)

#     def forward(self, x):
#         x = TF.functional.rgb_to_grayscale(x)
#         x1, x2, x3, x4, x5 = self.hed(x)

#         x5 = self.se_block5(x5)

#         x4 = x4 + F.interpolate(
#             x5, scale_factor=1, mode='bilinear', align_corners=True)
#         x4 = self.se_block4(x4)
#         x3 = x3 + F.interpolate(
#             x4, scale_factor=2, mode='bilinear', align_corners=True)
#         x3 = self.se_block3(x3)
#         x2 = x2 + F.interpolate(
#             x3, scale_factor=2, mode='bilinear', align_corners=True)
#         x2 = self.se_block2(x2)
#         x1 = x1 + F.interpolate(
#             x2, scale_factor=2, mode='bilinear', align_corners=True)
#         x1 = self.se_block1(x1)
#         x1 = self.conv_final(x1)

#         x1 = x1.view(x1.size(0), -1)

#         x1 = self.linear(x1)
#         x1 = torch.sigmoid(x1)
#         return x1


# class CustomDataset(Dataset):

#     def __init__(self, root_dir, cam_id, person_id, transform=None):
#         self.root_dir = root_dir
#         self.cam_id = cam_id
#         self.person_id = person_id
#         self.transform = transform
#         self.file_list = self._load_file_list()

#     def _load_file_list(self):
#         file_list = []
#         person_dir = os.path.join(self.root_dir, self.cam_id)
#         person_folder = os.path.join(person_dir, str(self.person_id).zfill(4))
#         if os.path.exists(person_folder):
#             for file in os.listdir(person_folder):
#                 if file.endswith('.jpg'):
#                     file_list.append(os.path.join(person_folder, file))
#         return file_list

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         image_path = self.file_list[idx]
#         image = Image.open(image_path)
#         if self.transform:
#             image = self.transform(image)
#         return image.cuda()


# transform = TF.Compose([
#     TF.CenterCrop((80, 80)),
#     TF.ToTensor(),
# ])


# def extract_features_and_cluster(network, dataset):
#     dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
#     features = []
#     for images in dataloader:
#         with torch.no_grad():
#             feature = network(images).cpu()
#             features.append(feature)
#     features = np.vstack(features)
#     kmeans = KMeans(n_clusters=1, random_state=0).fit(features)
#     return kmeans.cluster_centers_


# def calculate_distance(cluster1, cluster2):
#     return np.linalg.norm(cluster1 - cluster2)


# def train_network(network, dataset_path, epochs=1000):
#     optimizer = torch.optim.Adam(network.parameters())
#     for epoch in range(epochs):
#         total_loss = 0
#         for cam_id in ['cam2', 'cam6']:
#             cam_path = os.path.join(dataset_path, cam_id)
#             for person_id in os.listdir(cam_path):
#                 person_id = int(person_id)
#                 dataset = CustomDataset(dataset_path,
#                                         cam_id,
#                                         person_id,
#                                         transform=transform)
#                 if len(dataset) < 5:
#                     continue
#                 random_indices = random.sample(range(len(dataset)), 5)
#                 selected_dataset = torch.utils.data.Subset(
#                     dataset, random_indices)

#                 cluster_centers = extract_features_and_cluster(
#                     network, selected_dataset)

#                 if cam_id == 'cam2':
#                     cam1_centers = cluster_centers
#                 else:
#                     distance = calculate_distance(cam1_centers,
#                                                   cluster_centers)
#                     total_loss = total_loss + distance
#         loss = torch.tensor(total_loss, requires_grad=True).cuda()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
#     torch.save(network, 'edge_detection_model.pt')


# def main():
#     network = FeatureExtractor().cuda()
#     train_network(network, 'SYSU')


# if __name__ == "__main__":
#     main()
