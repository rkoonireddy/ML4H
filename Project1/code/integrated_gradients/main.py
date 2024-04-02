import os
import random
import sys
import time
from glob import glob

import cv2
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from dataset import PneumoniaDataset
from parameters import Parameters
import plotly.express as px
import matplotlib.pyplot as plt
from integrated_gradients import random_baseline_integrated_gradients
from visualization import visualize


def scale_pic(pic):
    return (pic - np.min(pic))/(np.max(pic) - np.min(pic))

def get_image_id_from_filename(image_name_string: str) -> str:
    substrings = image_name_string.split('\\')
    full_image_id = substrings[len(substrings) - 1] 
    image_id = full_image_id.split('.') 
    return image_id[0]

class GammaTransform:
    """Rotate by one of the given angles."""

    def __init__(self, gamma_min, gamma_max):
        self.gamma_range = [gamma_min, gamma_max]

    def __call__(self, input_image):
        return transforms.functional.adjust_gamma(input_image,
                                                  gamma=random.uniform(self.gamma_range[0],
                                                                       self.gamma_range[1]))

if __name__ == '__main__':
    # load cnn
    param = Parameters()
    net = resnet18(weights="DEFAULT")
    net.fc = torch.nn.Linear(512, 2)
    random.seed(111)
    
    # 5 healthy and 5 disease samples
    image_paths = []
    healthy_1 = 'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/NORMAL/IM-0007-0001.jpeg'
    healthy_2 = 'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/NORMAL/IM-0041-0001.jpeg'
    image_paths.append(healthy_1)
    image_paths.append(healthy_2)
  
    
    permanent_trans = transforms.Compose([
            transforms.Resize((224, 224), antialias=True)
    ])
   
    only_train_trans = transforms.RandomApply([
        transforms.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 0.2)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomAdjustSharpness(0.2)
        ], p=0.7)
  
    dataset = PneumoniaDataset(image_paths, permanent_trans, only_train_trans, reflection=True)
    dataloader = DataLoader(dataset, batch_size=param.batch_size, shuffle=True, num_workers=4)

    optimizer = None
    if param.optim_fcn == 'adam':
        optimizer = torch.optim.Adam([
            {'params': net.parameters()}
        ], lr=param.learning_rate, weight_decay=param.weight_decay)
    elif param.optim_fcn == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': net.parameters()}
        ], lr=param.learning_rate, weight_decay=param.weight_decay, momentum=0.9)
    elif param.optim_fcn == 'adagrad':
        optimizer = torch.optim.Adagrad([
            {'params': net.parameters()}
        ], lr=param.learning_rate, weight_decay=param.weight_decay)
    else:
        print('Wrong optim function!')
        sys.exit()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=param.scheduler_step_size, gamma=param.scheduler_gama)

    loss_function = None
    if param.loss_fcn == 'cross_entropy':
        loss_function = torch.nn.CrossEntropyLoss()
    elif param.loss_fcn == 'mse':
        loss_function = torch.nn.MSELoss()
    else:
        print('Wrong loss function!')
        sys.exit()

    #print(len(image_paths))
    images_ok = []
    for i in image_paths:
        img = cv2.imread(i)
        img = cv2.resize(img,(224,224))
        img = img.astype(np.double)
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img)
        images_ok.append(img)
    net = resnet18(weights="DEFAULT")
    net.fc = torch.nn.Linear(512, 2)
    net.load_state_dict(torch.load('./Project1/code/integrated_gradients/best_model.pth', map_location=torch.device('cuda')))

    attributions = random_baseline_integrated_gradients([images_ok[1]], net, steps=300, num_random_trials=2)
    img_integrated_gradient_overlay = visualize(attributions, images_ok[1], clip_above_percentile=99, clip_below_percentile=0,
                                                overlay=True, mask_mode=True)
    img_integrated_gradient = visualize(attributions, images_ok[1], clip_above_percentile=99, clip_below_percentile=0, overlay=False)
    """
    fig = px.imshow(images_ok[0].numpy().transpose(1, 2, 0))
    fig.show()
    fig2 = px.imshow(img_integrated_gradient_overlay.transpose(0, 1, 2))
    fig2.show()
    fig3 = px.imshow(img_integrated_gradient.transpose(0, 1, 2))
    fig3.show()
    """

    # Plotting the original image
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(scale_pic(images_ok[1].numpy().transpose(1, 2, 0)))
    plt.title('Original Image')
    plt.axis('off')

    # Plotting the image with integrated gradient overlay
    plt.subplot(1, 3, 2)
    plt.imshow(scale_pic(img_integrated_gradient_overlay.transpose(0, 1, 2)))
    plt.title('Integrated Gradient Overlay')
    plt.axis('off')

    # Plotting the image with integrated gradient
    plt.subplot(1, 3, 3)
    plt.imshow(scale_pic(img_integrated_gradient.transpose(0, 1, 2)))
    plt.title('Integrated Gradient')
    plt.axis('off')

    plt.show()


    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
