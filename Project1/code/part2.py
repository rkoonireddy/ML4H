import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# path to dataset
current_dir = os.getcwd()
dataset_path = os.path.join(current_dir, 'Project1/data')

# Explore the label distribution and qualitatively describe the data by plotting some examples for both labels. (1 Pt)
def explore_label_distribution(dataset_path):
    # there are 2 labels - NORMAL and PNEUMONIA
    
    num_normal_imgs_train = len(os.listdir(os.path.join(dataset_path, 'train/NORMAL')))
    num_pneumonia_imgs_train = len(os.listdir(os.path.join(dataset_path, 'train/PNEUMONIA')))

    """
    num_normal_imgs_val = len(os.listdir(os.path.join(dataset_path, 'val/NORMAL')))
    num_pneumonia_imgs_val = len(os.listdir(os.path.join(dataset_path, 'val/PNEUMONIA')))

    num_normal_imgs_test = len(os.listdir(os.path.join(dataset_path, 'test/NORMAL')))
    num_pneumonia_imgs_test = len(os.listdir(os.path.join(dataset_path, 'test/PNEUMONIA')))
    """
    print("---------------------------- TRAIN data ----------------------------")
    print('Number of images with label NORMAL: {} and with label PNEUMONIA: {}'.format(num_normal_imgs_train, num_pneumonia_imgs_train))
    """
    print("-------------------------- VALIDATION data --------------------------")
    print('Number of images with label NORMAL: {} and with label PNEUMONIA: {}'.format(num_normal_imgs_val, num_pneumonia_imgs_val))
    print("----------------------------- TEST data -----------------------------")
    print('Number of images with label NORMAL: {} and with label PNEUMONIA: {}'.format(num_normal_imgs_test, num_pneumonia_imgs_test))
    print('Label distribution on whole dataset: {} normal and {} pneumonia images'.format((num_normal_imgs_train + num_normal_imgs_val + num_normal_imgs_test)),(num_pneumonia_imgs_train + num_pneumonia_imgs_val + num_pneumonia_imgs_test))
    """

def plot_image_examples():
    #normal_img_path = os.path.join(dataset_path, 'train/NORMAL/')
    #img = os.listdir(normal_img_path)
    pass

explore_label_distribution(dataset_path)
