import argparse, os, sys, datetime, glob, importlib, csv
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


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb 
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt






logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        self.alpha_hat = self.alpha_hat.to(t.device)  # 디바이스 일치시키기
        # print("🏵️"*40)
        # print(t.device)
        # print(self.alpha_hat.device)
        
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            self.device = labels.device 
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                
                self.alpha = self.alpha.to(t.device)  # 디바이스 일치시키기
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                self.beta = self.beta.to(t.device)  # 디바이스 일치시키기
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


class ConditionalDiffusionModel(pl.LightningModule):
    def __init__(self, args):
        super(ConditionalDiffusionModel, self).__init__()
        self.args = args
        self.model = UNet_conditional(num_classes=args.num_classes)
        self.mse = nn.MSELoss()
        self.diffusion = Diffusion(img_size=args.image_size, device=self.device)
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

    def forward(self, x, t, labels):
        return self.model(x, t, labels)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        # print("🐥"*40)
        # print(images.device)
        # print(t.device)
        x_t, noise = self.diffusion.noise_images(images, t)
        predicted_noise = self.model(x_t, t, labels)
        loss = self.mse(noise, predicted_noise)
        self.ema.step_ema(self.ema_model, self.model)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(images, t)
        predicted_noise = self.model(x_t, t, labels)
        predicted_noise_ema = self.ema_model(x_t, t, labels) 
        val_loss = self.mse(noise, predicted_noise)
        val_loss_ema = self.mse(noise, predicted_noise_ema)
        self.log('val_loss_ema', val_loss_ema, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return val_loss
    
    def sample_images(self):
        device = self.device
        model = self.model
        diffusion = self.diffusion
        n = 8  # 샘플링할 이미지 수
        y = torch.Tensor([6] * n).long().to(device)  # 레이블 설정
        x = diffusion.sample(model, n, y, cfg_scale=0)
        return x


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def train_dataloader(self):
        return get_data(self.args)
    
    def val_dataloader(self):
        return val_get_data(self.args)
    
    # def on_train_epoch_start(self): 
    #     print("🎭"*20)
    #     print("       DDPM Train Start      ")
    #     print("🎭"*20)
    
    def on_train_epoch_end(self):
        # if (self.current_epoch + 1) % 10 == 0:
        print("🎸"*40)
        sampled_images = self.sample_images()
        # # 이미지 그리드 생성
        grid = torchvision.utils.make_grid(sampled_images, nrow=4)  # nrow는 그리드의 열 수
        # print(grid.size())
        # # 텐서를 NumPy 배열로 변환
        # grid = grid.permute(1, 2, 0).cpu().numpy()

        # # [0, 255] 범위로 스케일링 및 uint8 타입으로 변환
        # grid = (grid * 255).astype(np.uint8)
        

        # # 이미지 저장 경로 설정
        # save_dir = "saved_images"  # 이미지를 저장할 디렉토리
        # os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
        # file_name = f"epoch_{self.current_epoch}_grid.png"  # 파일 이름
        # file_path = os.path.join(save_dir, file_name)  # 전체 파일 경로

        # # 이미지 파일로 저장
        # plt.imsave(file_path, grid)
        
            # wandb에 이미지 그리드 로깅
        self.logger.experiment.log({"sampled_images": [wandb.Image(grid)]})
            # self.logger.experiment.log({"epoch": self.current_epoch, "sampled_images": [wandb.Image(grid)]})
            # self.logger.experiment.log({"sampled_images": [wandb.Image(grid)]}, step=self.current_epoch)
            # self.logger.experiment.log({"sampled_images": [wandb.Image(img) for img in sampled_images]})
    
    def on_train_end(self):
        torch.save(self.ema_model.state_dict(), "/media/hy/nwxxk/Diffusion-Models-pytorch-my/ema/ema_model.pt")










def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 100
    args.batch_size = 8
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = r"./datasets/Landscape_classifier_02/training"
    args.val_dataset_path = r"./datasets/Landscape_classifier_02/testing"
    args.device = "cuda"
    args.num_workers = 31
    args.lr = 3e-4
    return args
    # train(args)


def main(args):
    model = ConditionalDiffusionModel(args)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # logger = TensorBoardLogger("tb_logs", name="my_model")
    logger = WandbLogger(name=now, project="my_project")

    
    

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('checkpoints',now),
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min")
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,  # GPU 사용 설정 (0은 GPU 사용 안함)
        logger=logger,
        callbacks=[checkpoint_callback])

    trainer.fit(model)





if __name__ == '__main__':
    import torch
    torch.set_float32_matmul_precision('medium')  # 또는 'high'
    args = launch()
    main(args)

