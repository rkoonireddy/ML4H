import random
import os
import cv2
import numpy as np
import torch
from torchvision.models import resnet18
from parameters import Parameters
import matplotlib.pyplot as plt
from integrated_gradients import random_baseline_integrated_gradients
from visualization import visualize
from torchvision import transforms


def scale_pic(pic):
    return (pic - np.min(pic))/(np.max(pic) - np.min(pic))

def get_image_id_from_filename(image_name_string: str) -> str:
    substrings = image_name_string.split('\\')
    full_image_id = substrings[len(substrings) - 1] 
    image_id = full_image_id.split('.') 
    return image_id[0]

def get_image_id_from_filename2(image_name_string: str) -> str:
    substrings = image_name_string.split('/')
    full_image_id = substrings[len(substrings) - 1] 
    image_id = full_image_id.split('.') 
    return image_id[0]

def normalize(img):
    transform_norm = transforms.Compose([
        transforms.Normalize(mean=[122.7862, 122.7862, 122.7862], std=[60.2265, 60.2265, 60.2265])
    ])
    return transform_norm(img)

def resize(img):
    transform_resize = transforms.Compose([
        transforms.Resize((224, 224), antialias=True)
    ])
    return transform_resize(img)

if __name__ == '__main__':
    # load cnn
    param = Parameters()
    net = resnet18(weights="DEFAULT")
    net.fc = torch.nn.Linear(512, 2)
    random.seed(111)
    
    # 5 healthy and 5 disease samples
    image_paths = []
    """
    healthy_images_paths = ['C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/NORMAL/IM-0071-0001.jpeg',
                            'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/NORMAL/NORMAL2-IM-0297-0001.jpeg',
                            'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/NORMAL/IM-0041-0001.jpeg',
                            'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/NORMAL/NORMAL2-IM-0357-0001.jpeg',
                            'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/NORMAL/NORMAL2-IM-0378-0001.jpeg']
    disease_images_paths = ['C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/PNEUMONIA/person80_bacteria_391.jpeg',
                            'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/PNEUMONIA/person83_bacteria_407.jpeg',
                            'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/PNEUMONIA/person3_virus_17.jpeg',
                            'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/PNEUMONIA/person101_bacteria_483.jpeg',
                            'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/PNEUMONIA/person174_bacteria_832.jpeg']
    """
    healthy_images_paths = []
    disease_images_paths = []  
    show_mark = ['../Project1/data/chest_xray/test/PNEUMONIA/person108_bacteria_507.jpeg']
    
    for image_path in healthy_images_paths:
        image_paths.append(image_path)
    for image_path in disease_images_paths:
        image_paths.append(image_path)
    for image_path in show_mark:
        image_paths.append(image_path)

    #print(len(image_paths))
    images_ok = []
    images_names = []
    images_original = []
    for i in image_paths:
        img = cv2.imread(i)
        img = img.astype(np.double)
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img)
        img = img / 255.0
        images_original.append(img)

        ######## crop the image ########## 
        image_width = img.shape[1]
        image_height = img.shape[2]
        crop_transform = transforms.CenterCrop((int(image_width*0.75), int(image_height*0.75)))
        img = crop_transform(img)

        img = resize(img)
        #img = normalize(img)   
        images_ok.append(img)
        images_names.append(get_image_id_from_filename2(i))
    net = resnet18(weights="DEFAULT")
    net.fc = torch.nn.Linear(512, 2)
    net.load_state_dict(torch.load('../Project1/code/integrated_gradients/our_model_70p_random.pth', map_location=torch.device('cuda')))

    images_integrated_gradient_overlay = []
    images_integrated_gradient = []
    
    # plot
    if len(images_ok) > 1:
        fig, axes = plt.subplots(len(images_ok), 4, figsize=(20, 35))
        for cnt, img in enumerate(images_ok):
            
            attributions = random_baseline_integrated_gradients([img], net, steps=250, num_random_trials=15)
            img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0,
                                                overlay=True, mask_mode=True)
            #images_integrated_gradient_overlay.append(img_integrated_gradient_overlay)
            img_integrated_gradient = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
            #images_integrated_gradient.append(img_integrated_gradient)

            title = "Original - " + str(images_names[cnt])
            axes[cnt, 0].set_title(title)
            axes[cnt, 0].set_xlabel("Width [pixels]")
            axes[cnt, 0].set_ylabel("Height [pixels]")
            axes[cnt, 0].imshow(images_original[cnt].permute(1, 2, 0).numpy(), cmap='gray')
        
            title = "Cropped and resized - " + str(images_names[cnt])
            axes[cnt,1].set_title(title)
            axes[cnt,1].set_xlabel("Width [pixels]")
            axes[cnt,1].set_ylabel("Height [pixels]")
            axes[cnt,1].imshow(img.permute(1, 2, 0), cmap='gray')
    
            title = "Integrated gradient overlay - " + str(images_names[cnt])
            axes[cnt, 2].set_title(title)
            axes[cnt, 2].set_xlabel("Width [pixels]")
            axes[cnt, 2].set_ylabel("Height [pixels]")
            axes[cnt, 2].imshow(img_integrated_gradient_overlay.transpose(0, 1, 2), cmap='gray')

            title = "Integrated gradient - " + str(images_names[cnt])
            axes[cnt, 3].set_title(title)
            axes[cnt, 3].set_xlabel("Width [pixels]")
            axes[cnt, 3].set_ylabel("Height [pixels]")
            img_integrated_gradient = img_integrated_gradient / np.max(img_integrated_gradient)
            axes[cnt, 3].imshow(img_integrated_gradient.transpose(0, 1, 2), cmap='gray')
    
        plt.show()
        plt.close()
    
    else:
        fig, axes = plt.subplots(1, 4, figsize=(20, 35))
        for cnt, img in enumerate(images_ok):
            
            attributions = random_baseline_integrated_gradients([img], net, steps=250, num_random_trials=15)
            img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0,
                                                overlay=True, mask_mode=True)
            #images_integrated_gradient_overlay.append(img_integrated_gradient_overlay)
            img_integrated_gradient = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
            #images_integrated_gradient.append(img_integrated_gradient)

            title = "Original - " + str(images_names[cnt])
            axes[0].set_title(title)
            axes[0].set_xlabel("Width [pixels]")
            axes[0].set_ylabel("Height [pixels]")
            axes[0].imshow(images_original[cnt].permute(1, 2, 0).numpy(), cmap='gray')
        
            title = "Cropped and resized - " + str(images_names[cnt])
            axes[1].set_title(title)
            axes[1].set_xlabel("Width [pixels]")
            axes[1].set_ylabel("Height [pixels]")
            axes[1].imshow(img.permute(1, 2, 0), cmap='gray')
    
            title = "Integrated gradient overlay - " + str(images_names[cnt])
            axes[2].set_title(title)
            axes[2].set_xlabel("Width [pixels]")
            axes[2].set_ylabel("Height [pixels]")
            axes[2].imshow(img_integrated_gradient_overlay.transpose(0, 1, 2), cmap='gray')

            title = "Integrated gradient - " + str(images_names[cnt])
            axes[3].set_title(title)
            axes[3].set_xlabel("Width [pixels]")
            axes[3].set_ylabel("Height [pixels]")
            img_integrated_gradient = img_integrated_gradient / np.max(img_integrated_gradient)
            axes[3].imshow(img_integrated_gradient.transpose(0, 1, 2), cmap='gray')
    
        plt.show()
        plt.close()


    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()
