from glob import glob
import torch
from torch.utils.data import DataLoader
from dataset import PneumoniaDataset
from torchvision import transforms


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for n, (data, _) in enumerate(dataloader):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
        if n % 10 == 0:
            print('{}/{}'.format(n, len(dataloader)))

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


data_path = 'C:/Users/teodo/Desktop/repos/ML4H/Project1/data/chest_xray/*/*/*.jpeg'
x = glob(data_path)
permanent_trans = transforms.Compose([
            transforms.Resize((224, 224), antialias=True)
])
dataset = PneumoniaDataset(x,permanent_trans)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
means, stds = get_mean_and_std(dataloader)
print('{}, {}'.format(means, stds))
