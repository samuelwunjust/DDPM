import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self,beta_start=1e-2,beta_end=0.02,numsteps=1000,device='cuda',image_size=(512,512)):
        self.image_size=(512,512)
        self.device=device
        self.beta_start=beta_start
        self.beta_end=beta_end
        self.numsteps=numsteps
        self.beta=self.create_noise_schedule().to(device)
        self.alpha=1-self.beta
        self.alpha_hat=torch.cumprod(self.alpha,dim=0)
    def create_noise_schedule(self):
        return torch.linspace(self.beta_start,self.beta_end,steps=self.numsteps)

    def noise_image(self,x,t):#xt=sqrt(αt_hat)x0+sqrt(1-αt_hat)*noise
        
        x0=x
        coffie0=torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        coffien=torch.sqrt(1-self.alpha_hat[t])[:,None,None,None]
        noise=torch.randn_like(x)     
        return coffie0*x0+coffien*noise,noise
    
    def sample_time_steps(self,n):
        return torch.randint(low=1,high=self.numsteps,size=(n,))
    
    def sample(self,model,n,labels,clg_scale=3):#推理过程
        logging.info(f"Sampling {n} new images")
        model.eval()#转换为评估模式
        print(f"we will sample {n}images")
        with torch.no_grad():
            x=torch.randn((n,3,self.img_))
            for i in tqdm(reversed(range(1,self.numsteps)),position=0):
                t=(torch.ones(x)*i).long().to(self.device)
                predicted_noise=model(x,t,labels)
                if clg_scale>0:
                    uncon_predicted_noise=model(x,t,None)
                    predicted_noise=torch.lerp(uncon_predicted_noise,predicted_noise,clg_scale)
                alpha=self.alpha[t][:,None,None,None]
                alpha_hat=self.alpha_hat[t][:,None,None,None]
                beta=self.beta[t][:None,None,None]
                if i>1:
                    noise=torch.randn_like(x)
                else:
                    noise=torch.zeros_like(x)
                
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()    
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(args):
        setup_logging(args.run_name)
        device=args.device
        dataloader=get_data(args)
        mseloss=nn.MSELoss()
        model=UNet_conditional(num_classes=args.num_classes).to(device)
        optimizer=optim.Adam(model.parameters(),lr=args.lr)
        diffusion=Diffusion(image_size=args.image_size,device=device)
        logger = SummaryWriter(os.path.join("runs", args.run_name))
        l = len(dataloader)
        ema=EMA(0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        
        for epoch in range(args.epochs):
             logging.info(f"train {epoch}")
             pb=tqdm(dataloader)
             for i,(image,labels) in enumerate(pb):
                image=image.to(device)
                labels=labels.to(device)
                t=diffusion.sample_time_steps(image.shape[0]).to(device)
                
                x_t,noise=diffusion.noise_image(image,t)
                if np.random.random()<0.1:
                    labels=None
                predicted_noise=model(x_t,t,labels)
                loss=mseloss(predicted_noise,noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.step_ema(ema_model, model)

                pb.set_postfix(MSE=loss.item())
                logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            
             if epoch % 10 == 0:
              labels = torch.arange(10).long().to(device)
              sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
              ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
              plot_images(sampled_images)
              save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
              save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
              torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
              torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
              torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 15
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = "./cifar10/cifar10-64/train"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
    device = "cuda"
    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(image_size=64, device=device)
    n = 8
    y = torch.Tensor([6] * n).long().to(device)
    x = diffusion.sample(model, n, y, cfg_scale=0)
    plot_images(x)



                









    
