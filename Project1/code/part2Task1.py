import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# path to dataset
current_dir = os.getcwd()
dataset_path = os.path.join(current_dir, 'Project1/data/chest_xray')

# Explore the label distribution and qualitatively describe the data by plotting some examples for both labels. (1 Pt)
def explore_label_distribution(dataset_path):
    # there are 2 labels - NORMAL and PNEUMONIA
    
    num_normal_imgs_train = len(os.listdir(os.path.join(dataset_path, 'train/NORMAL')))
    num_pneumonia_imgs_train = len(os.listdir(os.path.join(dataset_path, 'train/PNEUMONIA')))

    num_normal_imgs_val = len(os.listdir(os.path.join(dataset_path, 'val/NORMAL')))
    num_pneumonia_imgs_val = len(os.listdir(os.path.join(dataset_path, 'val/PNEUMONIA')))
 
    num_normal_imgs_test = len(os.listdir(os.path.join(dataset_path, 'test/NORMAL')))
    num_pneumonia_imgs_test = len(os.listdir(os.path.join(dataset_path, 'test/PNEUMONIA')))
    
    print("---------------------------- TRAIN data ----------------------------")
    print('Number of images with label NORMAL: {} and with label PNEUMONIA: {}'.format(num_normal_imgs_train, num_pneumonia_imgs_train))

    print("-------------------------- VALIDATION data --------------------------")
    print('Number of images with label NORMAL: {} and with label PNEUMONIA: {}'.format(num_normal_imgs_val, num_pneumonia_imgs_val))
    
    print("----------------------------- TEST data -----------------------------")
    print('Number of images with label NORMAL: {} and with label PNEUMONIA: {}'.format(num_normal_imgs_test, num_pneumonia_imgs_test))
    print('Label distribution on whole dataset: {} normal and {} pneumonia images'.format(num_normal_imgs_train + num_normal_imgs_val + num_normal_imgs_test,num_pneumonia_imgs_train + num_pneumonia_imgs_val + num_pneumonia_imgs_test))
    

def plot_image_examples():
    # plot easy visual differences for bacterial pneumonia
    easy_bacteria_1_path = os.path.join(dataset_path, 'train/PNEUMONIA/person1022_bacteria_2953.jpeg')
    easy_bacteria_2_path = os.path.join(dataset_path, 'test/PNEUMONIA/person95_bacteria_463.jpeg')
    
    img1 = mpimg.imread(easy_bacteria_1_path)
    img2 = mpimg.imread(easy_bacteria_2_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Plot first image
    axes[0].set_title("Bacterial pneumonia")
    axes[0].set_xlabel("Width [pixels]")
    axes[0].set_ylabel("Height [pixels]")
    axes[0].imshow(img1, cmap='gray')
    # Plot second image
    axes[1].set_title("Bacterial pneumonia")
    axes[1].set_xlabel("Width [pixels]")
    axes[1].set_ylabel("Height [pixels]")
    axes[1].imshow(img2, cmap='gray')
    plt.show()
    ################################################################################################
    # plot significantly progressed pneumonia
    easy_normal_1_path = os.path.join(dataset_path, 'train/NORMAL/IM-0147-0001.jpeg')
    easy_virus_pneumonia_1_path = os.path.join(dataset_path, 'train/PNEUMONIA/person1015_virus_1701.jpeg')
    easy_bacteria_pneumonia_1_path = os.path.join(dataset_path, 'train/PNEUMONIA/person255_bacteria_1165.jpeg')

    img1 = mpimg.imread(easy_normal_1_path)
    img2 = mpimg.imread(easy_virus_pneumonia_1_path)
    img3 = mpimg.imread(easy_bacteria_pneumonia_1_path)
    #max_width = max(img1.shape[1], img2.shape[1], img3.shape[1])
    #max_height = max(img1.shape[0], img2.shape[0], img3.shape[0])
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    # Plot first image
    axes[0].set_title("Healthy sample")
    axes[0].set_xlabel("Width [pixels]")
    axes[0].set_ylabel("Height [pixels]")
    #axes[0].imshow(img1, cmap='gray', extent=[0, max_width, max_height, 0])
    axes[0].imshow(img1, cmap='gray')
    # Plot second image
    axes[1].set_title("Virus pneumonia")
    axes[1].set_xlabel("Width [pixels]")
    axes[1].set_ylabel("Height [pixels]")
    axes[1].imshow(img2, cmap='gray')
    # Plot third image
    axes[2].set_title("Bacterial pneumonia")
    axes[2].set_xlabel("Width [pixels]")
    axes[2].set_ylabel("Height [pixels]")
    axes[2].imshow(img3, cmap='gray')
    plt.show()

    ###################################################################################################
    # no clear differences to untrained eye
    hard_normal_path = os.path.join(dataset_path, 'test/NORMAL/IM-0049-0001.jpeg.')
    hard_pneumonia_virus_path = os.path.join(dataset_path, 'test/PNEUMONIA/person79_virus_148.jpeg')
    hard_pneumonia_bacteria_path = os.path.join(dataset_path, 'test/PNEUMONIA/person91_bacteria_445.jpeg')
    
    img1 = mpimg.imread(hard_normal_path)
    img2 = mpimg.imread(hard_pneumonia_virus_path)
    img3 = mpimg.imread(hard_pneumonia_bacteria_path)
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    # Plot first image
    axes[0].set_title("Healthy sample")
    axes[0].set_xlabel("Width [pixels]")
    axes[0].set_ylabel("Height [pixels]")
    axes[0].imshow(img1, cmap='gray')
    # Plot second image
    axes[1].set_title("Virus pneumonia")
    axes[1].set_xlabel("Width [pixels]")
    axes[1].set_ylabel("Height [pixels]")
    axes[1].imshow(img2, cmap='gray')
    # Plot third image
    axes[2].set_title("Bacterial pneumonia")
    axes[2].set_xlabel("Width [pixels]")
    axes[2].set_ylabel("Height [pixels]")
    axes[2].imshow(img3, cmap='gray')
    plt.show()


explore_label_distribution(dataset_path)
plot_image_examples()


