# ML4H
Machine Learning for Health Care

Part 2: Pneumonia Prediction Dataset

Data is placed in folders with relative paths:
- Project1\data\chest_xray\train\PNEUMONIA
- Project1\data\chest_xray\train\NORMAL
- Project1\data\chest_xray\val\PNEUMONIA
- Project1\data\chest_xray\val\NORMAL
- Project1\data\chest_xray\test\PNEUMONIA
- Project1\data\chest_xray\test\NORMAL

Q1: To run Exploratory Data Analysis run Project1\code\part2Task1.py. This script includes plotting some samples. 

Q2: Everything connected with this task is in folder Project1\code\CNN. Most important file is main.py which calls train function.

Q3: Everything connected with this task is in folder Project1\code\integrated_gradients. CNN model that performed best is renamed to our_model.pth. To run the code and see plotted images, run Project1\code\integrated_gradients\main.py.

Q5: 
1. Make sure to change line 135 in Project1\code\CNN\main.py --> dataset_train = PneumoniaDataset(training_list, permanent_trans, only_train_trans, reflection=False, randomize_labels=False, randomization_percentage=0.7) change randomize_labels=False to randomize_labels=True
2. If you want to visualize part with integrated gradients make sure to change line 102 in Project1\code\integrated_gradients\main.py --> net.load_state_dict(torch.load('./Project1/code/integrated_gradients/our_model.pth', map_location=torch.device('cuda'))) to   net.load_state_dict(torch.load('./Project1/code/integrated_gradients/our_model_70p_random.pth', map_location=torch.device('cuda')))

! Be aware of the problem with 1 dot and 2 dots: example: './Project1/data/chest_xray/test/PNEUMONIA/person80_bacteria_391.jpeg' sometimes need to be changed to '../Project1/data/chest_xray/test/PNEUMONIA/person80_bacteria_391.jpeg' and vice versa