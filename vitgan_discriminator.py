import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import vitgan_generator
import Edge_extract
import numpy as np
from PIL import Image
from Edge_extract import SobelEdgeDetection


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
])


edge_detection_model = torch.load("edge_detection_model.pt").to(device)


generator = vitgan_generator.GeneratorViT(
        style_mlp_layers=8,
        patch_size=4,
        latent_dim=32,
        hidden_size=384,
        image_size=80,
        depth=4,
        combine_patch_embeddings=False,
        combined_embedding_size=1024,
        forward_drop_p=0.0,
        bias=False,
        out_features=3,
        out_patch_size=4,
        weight_modulation=True,
        siren_hidden_layers=1
    )
generator=generator.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(generator.parameters(),lr=0.00001)


def train(epochs=100):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(1, 5):
            cam1_dataset = Edge_extract.CustomDataset("SYSU", "cam1", i, transform=transform)
            cam1_image = cam1_dataset[0].unsqueeze(0).to(device)
            x1 = edge_detection_model(cam1_image)

            z = x1
            output = generator(z)
            output = output.permute(0, 2, 1).contiguous()
            img = output.view(1, 3,80,80)
            x2 = edge_detection_model(img)

      
            cam6_dataset = Edge_extract.CustomDataset("SYSU", "cam6", i, transform=transform)
            cam6_image = cam6_dataset[0].unsqueeze(0).to(device)
            x3 = edge_detection_model(cam6_image)

    
            loss_cam1 = criterion(x2, x1)
            loss_cam6 = criterion(x2, x3)
            loss = -(loss_cam1 + loss_cam6)
            total_loss+=loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    torch.save(generator, "generator.pt")

def main():
    train()

if __name__=="__main__":
    train()
