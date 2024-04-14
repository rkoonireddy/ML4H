from captum.attr import IntegratedGradients
import torch
from torchvision.models import resnet18
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 5 healthy and 5 disease samples
image_paths = []
healthy_1 = 'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/NORMAL/IM-0007-0001.jpeg'
healthy_2 = 'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/test/NORMAL/IM-0041-0001.jpeg'
image_paths.append(healthy_1)
image_paths.append(healthy_2)

images_ok = []
for i in image_paths:
    img = cv2.imread(i)
    img = cv2.resize(img,(224,224))
    img = img.astype(np.float32) / 255.0  # Normalize the image to [0, 1] range
    img = img.transpose(2,0,1)
    img = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension
    images_ok.append(img)

input1 = images_ok[0].clone().requires_grad_(True)  # Clone the tensor to avoid modification
# Initializing our toy model
model = resnet18(weights="DEFAULT")  # Using pretrained weights
model.fc = torch.nn.Linear(512, 2)  # Adjusting the last layer for your binary classification
model.load_state_dict(torch.load('./Project1/code/integrated_gradients/best_model.pth'))
model.eval()  # Setting model to evaluation mode
# Applying integrated gradients on the input
ig = IntegratedGradients(model)
target = torch.tensor([0])  # Index 0 for the first output
input1_attr, delta = ig.attribute(input1, n_steps=100, target=target, return_convergence_delta=True)

# Convert attribution tensor to numpy array and squeeze batch dimension
attribution = input1_attr.squeeze(0).cpu().detach().numpy()

# Normalize attribution values to [0, 1] for visualization
attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())

# Convert the original image tensor to numpy and transpose it to (H, W, C) format
original_image = input1.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)

# Resize attribution map to match the original image dimensions
attribution_resized = cv2.resize(attribution, (original_image.shape[1], original_image.shape[0]))

# Plot the original image and the attribution map
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(attribution_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
axes[1].set_title('Attribution Map')
axes[1].axis('off')

plt.show()