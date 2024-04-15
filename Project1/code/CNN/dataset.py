from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


def get_image_id_from_filename(image_name_string: str) -> str:
    substrings = image_name_string.split('\\')
    full_image_id = substrings[len(substrings) - 1] 
    image_id = full_image_id.split('.')
    return image_id[0]

class PneumoniaDataset(Dataset):
    def __init__(self, data, transform, train_transform=None, reflection=False):
        self.data = np.asarray(data)
        self.resize = transform
        self.train_transform = train_transform
        self.reflection = reflection

    def normalize(self, img):
        transform_norm = transforms.Compose([
            transforms.Normalize(mean=[122.7862, 122.7862, 122.7862], std=[60.2265, 60.2265, 60.2265])
        ])
        return transform_norm(img)

    def __getitem__(self, index):
        file = self.data[index]
        #  file = self.data.iloc[index, 0]
        #  file = 'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray\train\PNEUMONIA\person1294_virus_2221.jpeg'
        
        ######### read the image ########
        img = cv2.imread(file)
        #img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) 
        img = img.transpose(2, 1, 0) 
        img = torch.from_numpy(img)
        #img = img / 255.0
        
        ######## crop the image ##########
       
        #print(img.shape)  # (3, width, height)
        image_width = img.shape[1]
        image_height = img.shape[2]
        crop_transform = transforms.CenterCrop((int(image_width*0.75), int(image_height*0.75)))
        img = crop_transform(img)

        ######## resize the image ########
        image_resized = self.resize(img)
       
        ######## apply transformations #########
        image_transformed = image_resized
        if self.train_transform is not None:
            image_transformed = self.train_transform(image_resized)

        """
        # plot differences after transformation
        fig, axes = plt.subplots(1, 4, figsize=(28, 5))
        # Plot original image
        title = get_image_id_from_filename(file) + str(" - original")
        axes[0].set_title(title)
        axes[0].set_xlabel("Width [pixels]")
        axes[0].set_ylabel("Height [pixels]")
        #image_original = np.clip(image_original, 0, 1)
        axes[0].imshow(img.permute(2, 1, 0).numpy(), cmap='gray')
        # Plot cropped image
        title = get_image_id_from_filename(file) + str(" - cropped")
        axes[1].set_title(title)
        axes[1].set_xlabel("Width [pixels]")
        axes[1].set_ylabel("Height [pixels]")
        #image_cropped = np.clip(image_cropped, 0, 1)
        axes[1].imshow(img_cropped.permute(2, 1, 0).numpy(), cmap='gray')
        #plot resized image
        title = get_image_id_from_filename(file) + str(" - resized")
        axes[2].set_title(title)
        axes[2].set_xlabel("Width [pixels]")
        axes[2].set_ylabel("Height [pixels]")
        #image_resized = np.clip(image_resized, 0, 1)
        axes[2].imshow(image_resized.permute(2, 1, 0).numpy(), cmap='gray')
        #plot transformed image
        title = get_image_id_from_filename(file) + str(" - transformed")
        axes[3].set_title(title)
        axes[3].set_xlabel("Width [pixels]")
        axes[3].set_ylabel("Height [pixels]")
        #image_transformed = np.clip(image_transformed, 0, 1)
        axes[3].imshow(image_transformed.permute(2, 1, 0).numpy(), cmap='gray')
        plt.show()
        plt.close()
        """

        ###### adding noise to image #######
        ######### normalize the image ########
        #img = np.transpose(img, (1, 2, 0))
        #mean, std = img.mean([0,1,2]), img.std([0,1,2])
        image_transformed = self.normalize(image_transformed)
        noise = torch.zeros(3, 224, 224, dtype=torch.float)
        noise = noise + (0.001 ** 0.5) * torch.randn(3, 224, 224)
        img_final = image_transformed + noise

        if torch.isnan(img_final).any():
            print('Image has nan')
        target = int((1 if 'PNEUMONIA' in file else 0))
        return img_final, target

    def __len__(self):
        return len(self.data)

