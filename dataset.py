from torch.utils import data
import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

class MNIST(data.Dataset):
    def __init__(self, root="./data", transform=None):
        self.root = root
        self.transform = transform
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.file = os.listdir(self.root)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.file[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)

        return image
    
    def __len__(self):
        return len(self.file)

if __name__ == '__main__':
    dataset = MNIST()
    print(dataset[0])
    save_image(dataset[0], "test.png")
