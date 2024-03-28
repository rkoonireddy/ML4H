import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas
from PIL import Image
import matplotlib.pyplot as plt

def get_image_id_from_filename(image_name_string: str) -> str:
    substrings = image_name_string.split('\\')
    full_image_id = substrings[len(substrings) - 1] 
    image_id = full_image_id.split('.')
    return image_id[0]

class PneumoniaDataset(Dataset):
    def __init__(self, data, transform, train_transform=None, reflection=False):
        self.data = np.asarray(data)
        self.transform = transform
        self.train_transform = train_transform
        self.reflection = reflection
        self.normalization = transforms.Compose([
            transforms.Normalize(mean=[275.1504, 275.1504, 275.1504], std=[370.7895, 370.7895, 370.7895])
        ])

    def __getitem__(self, index):
        file = self.data[index]
        #  file = self.data.iloc[index, 0]
        #  file = 'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray\train\PNEUMONIA\person1294_virus_2221.jpeg'
        image = Image.open(file)
        image_rbg = image.convert('RGB')
        #image_tensor =  transforms.ToTensor()(image_rbg).transpose(2, 1, 0)       
        image_tensor =  transforms.ToTensor()(image_rbg) 
        #image_original = torch.tensor(image_tensor, dtype=torch.float)
        image_original = image_tensor.clone().detach()
        #print(image_original)
        
        #print(image_original.shape)  # (3, height, width)
        image_height = image_original.shape[1]
        image_width = image_original.shape[2]
        crop_transform = transforms.CenterCrop((image_height-60, image_width-260))
        image_cropped = crop_transform(image_original)
        image_cropped = image_cropped
        #print("cropped")
        image_resized = self.transform(image_cropped)
        #print("resized")
        image_transformed = image_resized
        if self.train_transform is not None:
            image_transformed = self.train_transform(image_resized)
        #print("transformed")
        # plot differences after transformation
        fig, axes = plt.subplots(1, 4, figsize=(28, 5))
        # Plot original image
        title = get_image_id_from_filename(file) + str(" - original")
        axes[0].set_title(title)
        axes[0].set_xlabel("Width [pixels]")
        axes[0].set_ylabel("Height [pixels]")
        image_original = np.clip(image_original, 0, 1)
        axes[0].imshow(image_original.permute(1, 2, 0).numpy(), cmap='gray')
        # Plot cropped image
        title = get_image_id_from_filename(file) + str(" - cropped")
        axes[1].set_title(title)
        axes[1].set_xlabel("Width [pixels]")
        axes[1].set_ylabel("Height [pixels]")
        image_cropped = np.clip(image_cropped, 0, 1)
        axes[1].imshow(image_cropped.permute(1, 2, 0).numpy(), cmap='gray')
        #plot resized image
        title = get_image_id_from_filename(file) + str(" - resized")
        axes[2].set_title(title)
        axes[2].set_xlabel("Width [pixels]")
        axes[2].set_ylabel("Height [pixels]")
        image_resized = np.clip(image_resized, 0, 1)
        axes[2].imshow(image_resized.permute(1, 2, 0).numpy(), cmap='gray')
        #plot transformed image
        title = get_image_id_from_filename(file) + str(" - transformed")
        axes[3].set_title(title)
        axes[3].set_xlabel("Width [pixels]")
        axes[3].set_ylabel("Height [pixels]")
        image_transformed = np.clip(image_transformed, 0, 1)
        axes[3].imshow(image_transformed.permute(1, 2, 0).numpy(), cmap='gray')
        #plt.show()
        plt.close()

        image = self.normalization(image_transformed)
        noise = torch.zeros(3, 224, 224, dtype=torch.float)
        noise = noise + (0.001 ** 0.5) * torch.randn(3, 224, 224)
        image = image + noise

        if torch.isnan(image).any():
            print('Image has nan')
        target = int((1 if 'PNEUMONIA' in file else 0))
        return image, target

    def __len__(self):
        return len(self.data)

