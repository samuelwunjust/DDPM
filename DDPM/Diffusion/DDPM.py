import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
class diffusion:
    def __init__(self,numsteps=1000,beta_start=1e-2,beta_end=0.02,imgsize=(512,512),device='cuda'):
       self.numsteps=numsteps
       self.beta_start=beta_start
       self.beta_end=beta_end
       self.imgsize=imgsize
       self.device=self.device
       self.beta=self.prepare_noise_scheduler().to(device)
       self.alpha=1-self.beta
       self.alpha_hat=torch.cumprod(self.alpha,dim=0)

    def create_noise_scheduler(self):
        return torch.linspace(self.beta_start,self.beta_end,self.numsteps)
    
    def noise_image(self,x,t):#xt=sqrt(αt_hat)x0+sqrt(1-αt_hat)*noise
        sqrt_alpha_hat=torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        sqrt_one_minus_alpha_hat=torch.sqrt(1-sqrt_alpha_hat)
        noise=torch.randn_like(x)
        return sqrt_alpha_hat*x+sqrt_one_minus_alpha_hat*noise,noise

    def sample_time_steps(self,n):
        return torch.randint(low=1,high=self.sample_time_steps,size=(n,))
    
    def sample(self,model,n):#推理过程
        logging.info(f"Sampling {n} new images")
        model.eval()#转换为评估模式
        print(f"we will sample {n}images")
        with torch.no_grad():
            x=torch.randn((n,3,self.imgsize,self.imgsize)).long().to(self.device)
            for i in tqdm(reversed(range(1,self.sample_time_steps)),position=0):
                t=(torch.ones(n)*i).long().to(self.device)
                predicted_noise=model(x,t)
                alpha=self.alpha[t][:,None,None,None]
                alpha_hat=self.alpha_hat[t][:,None,None,None]
                beta=self.beta[t][:,None,None,None]

                if i>1:
                    noise=torch.randn_like(x)
                else:
                    noise=torch.zeros_like(x)
                
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()    
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(self,args):
        setup_logging(args.run_name)
        device=args.device
        dataloader=get_data(args)
        mseloss=nn.MSELoss()
        model=UNet().to(device)
        optimizer=optim.Adam(model.parameters(),lr=args.lr)
        diffusion=diffusion(imgsize=args.image_size,device=device)
        logger = SummaryWriter(os.path.join("runs", args.run_name))
        l = len(dataloader)
        for epoch in range(args.epochs):
             
             logging.info(f"Starting epoch {epoch}:")
             pb=tqdm(dataloader)
             for i,(image,) in enumerate(pb):
                image=image.to(device)
                t=diffusion.sample_timesteps(image.shape[0]).to(device)
                x_t,noise=diffusion.noise_image(image,t)
                predicetd_noise=model(x_t,t)
                loss=mseloss(predicetd_noise,noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pb.set_postfix(MSE=loss.item())
                logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        sampled_images = diffusion.sample(model, n=image.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

                
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()

        
