import os
import random
import sys
import time
from glob import glob

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from dataset import PneumoniaDataset
from parameters import Parameters
from training import train

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
    print(os.getcwd())
    save_path = f'./code/CNN/results/Run_{time.strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(save_path)
    param = Parameters()
    param.save(save_path, 'Parameters')
    net = resnet18(weights="DEFAULT")
    net.fc = torch.nn.Linear(512, 2)
    #  net2d.load_state_dict(torch.load('./results/Run_20240329_155858/model_15.pth'))
    random.seed(111)

    data_path = './Project1/data/chest_xray/*/*/*.jpeg'
    #  x = np.asarray(range(len(glob(data_path))))
    x = glob(data_path)
    x_no_test = [file for file in x if 'test' not in file]
    x_test = [file for file in x if 'test' in file]
    pneumonia_list_ids = []
    normal_list_ids = []
    for filename in x_no_test:
        image_id = get_image_id_from_filename(filename)
        if 'PNEUMONIA' in filename and len(pneumonia_list_ids) < 1350:
            if len(pneumonia_list_ids) == 0 or pneumonia_list_ids[len(pneumonia_list_ids) - 1] != image_id:
                pneumonia_list_ids.append(image_id)
        else:
            if len(normal_list_ids) == 0 or normal_list_ids[len(normal_list_ids) - 1] != image_id:
                normal_list_ids.append(image_id)
    random.shuffle(pneumonia_list_ids)
    random.shuffle(normal_list_ids)

    train_pneumonia_list_ids = pneumonia_list_ids[:int(0.8*len(pneumonia_list_ids))]
    validation_pneumonia_list_ids = pneumonia_list_ids[int(0.8 * len(pneumonia_list_ids)):]
    train_normal_list_ids = normal_list_ids[:int(0.8 * len(normal_list_ids))]
    validation_normal_list_ids = normal_list_ids[int(0.8 * len(normal_list_ids)):]
    
    pneumonia_list_ids = []
    normal_list_ids = []
    for filename in x_test:
        image_id = get_image_id_from_filename(filename)
        if 'PNEUMONIA' in filename:
            if len(pneumonia_list_ids) == 0 or pneumonia_list_ids[len(pneumonia_list_ids) - 1] != image_id:
                pneumonia_list_ids.append(image_id)
        else:
            if len(normal_list_ids) == 0 or normal_list_ids[len(normal_list_ids) - 1] != image_id:
                normal_list_ids.append(image_id)
    random.shuffle(pneumonia_list_ids)
    random.shuffle(normal_list_ids)

    test_pneumonia_list_ids = pneumonia_list_ids
    test_normal_list_ids = normal_list_ids

    train_pneumonia_images = []
    validation_pneumonia_images = []
    test_pneumonia_images = []
    train_normal_images = []
    validation_normal_images = []
    test_normal_images = []

    file_cnt = 0
    for filename in x:
        file_cnt += 1
        img_id = get_image_id_from_filename(filename)
        if 'PNEUMONIA' in filename:
            if img_id in train_pneumonia_list_ids:
                train_pneumonia_images.append(filename)
            if img_id in validation_pneumonia_list_ids:
                validation_pneumonia_images.append(filename)
            if img_id in test_pneumonia_list_ids:
                test_pneumonia_images.append(filename)
        if 'NORMAL' in filename:
            if img_id in train_normal_list_ids:
                train_normal_images.append(filename)
            if img_id in validation_normal_list_ids:
                validation_normal_images.append(filename)
            if img_id in test_normal_list_ids:
                test_normal_images.append(filename)

    # pneumonia, normal = count_classes(x[:train_proc])
    print('Training - {} pneumonia and {} normal'.format(len(train_pneumonia_images), len(train_normal_images)))
    # pneumonia, normal = count_classes(x[train_proc:(train_proc + val_proc)])
    print('Validation - {} pneumonia and {} normal'.format(len(validation_pneumonia_images), len(validation_normal_images)))
    # pneumonia, normal = count_classes(x[(train_proc + val_proc):])
    print('Test - {} pneumonia and {} normal'.format(len(test_pneumonia_images), len(test_normal_images)))

    training_list = train_pneumonia_images + train_normal_images
    validation_list = validation_pneumonia_images + validation_normal_images
    test_list = test_pneumonia_images + test_normal_images
   
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
    
    #chage randomization_label flag and percentage
    dataset_train = PneumoniaDataset(training_list, permanent_trans, only_train_trans, reflection=False,randomize_labels=False,randomization_percentage=0.7) 
    dataset_train.save_to_excel(save_path, 'dataset_parameters')
    dataloader_train = DataLoader(dataset_train, batch_size=param.batch_size, shuffle=True, num_workers=4)
    dataset_validation = PneumoniaDataset(validation_list, permanent_trans)
    dataloader_validation = DataLoader(dataset_validation, batch_size=param.batch_size, shuffle=True, num_workers=4)
    dataset_test = PneumoniaDataset(test_list, permanent_trans)
    dataloader_test = DataLoader(dataset_test, batch_size=param.batch_size, shuffle=False, num_workers=4)

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

    train(net, dataloader_train, dataloader_validation, dataloader_test, optimizer, param.num_of_epochs,
          loss_function, scheduler,
          save_path)
