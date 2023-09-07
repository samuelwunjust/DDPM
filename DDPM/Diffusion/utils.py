import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([torch.cat([i for i in images.cpu()], dim=-1),], dim=-2).permute(1, 2, 0).cpu())
    plt.show()



def save_images(image,path,**kwargs):
    grid=torchvision.utils.make_grid(image,**kwargs)
    narrd=grid.permute(1,2,0).to("cpu").numpy()
    im=Image.fromarray(narrd)
    im.save(path)

def get_data(args):
    transform=torchvision.transforms.Compose(
           [torchvision.transforms.Resize(80),
            torchvision.transforms.RandomResizedCrop(args.image_size,scale=(0.8,1)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]

    )
    dataset=torchvision.datasets.ImageFolder(args.dataset_path,transform=transform)
    dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False)
    return dataloader

def setup_logging(runname):
    os.makedirs('models',exist_ok=True)
    os.makedirs('results',exist_ok=True)
    os.makedirs(os.path.join('models',runname),exist_ok=True)
    os.makedirs(os.path.join('results',runname),exist_ok=True)
    









