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