import sys, datetime, time
import numpy as np
from pathlib import Path
import torch
import torchvision
from PIL import Image
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch.nn as nn
import lightning as L


from modules.loss import LPIPS_MAE_klloss
from models.cvae import CVAE
from data.get_cifar100 import get_data
from data.cifar_10 import get_cifar10_data

class SetupCallback(Callback):
    def __init__(self, now,logdir):
        super().__init__()
        self.logdir=Path(logdir)
        self.now = now
        self.ckptdir=self.logdir/'checkpoints'
    def on_exception(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = self.ckptdir / "last.ckpt"
            trainer.save_checkpoint(str(ckpt_path))

    def setup(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            self.logdir.mkdir(parents=True, exist_ok=True)
            self.ckptdir.mkdir(parents=True, exist_ok=True)

class ImageLogger(Callback):
    def __init__(self, save_dir,batch_frequency, max_images, image_enhence=1.0,clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.save_dir=save_dir
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.imgae_enhance=image_enhence
        self.logger_log_images = {
            TensorBoardLogger:self._testtube
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, split, images,
                  global_step, current_epoch, batch_idx):
        root = Path(self.save_dir)/'images'/split
        
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "gs-{:06}_e-{:06}_b-{:06}.png".format(
                
                global_step,
                current_epoch,
                batch_idx)
            path = root/k/filename
            
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            image = Image.fromarray(grid)
            image.save(path)

    def log_img(self, pl_module, batch, batch_idx, split):
       
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx,split) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local( split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx,split):
        if split=='train':
            batch_freq=self.batch_freq
        elif split=='val':
            batch_freq=self.batch_freq/5
        if ((check_idx % batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx='train_loader'):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx='val_loader'):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    sys.path.append('./cVAE')
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir=Path('logs')
    sys.path.append(Path.cwd())
    lr_callback=LearningRateMonitor(logging_interval='step')
    loss=LPIPS_MAE_klloss()
    ch_mult=[8,16]
    mask_rate=0.5
    name=now+str(ch_mult)+'klweight'+str(0.00075)
    ts_logger=TensorBoardLogger(save_dir=logdir,name=name,version=0)
    img_logger=ImageLogger(save_dir=str(logdir/name/'images'),batch_frequency=120,max_images=8,log_on_batch_idx=True)
    model_ckpt_callback=ModelCheckpoint( 
                    dirpath=str(logdir/name/'checkpoints'),
                    monitor='val_loss',
                    verbose= True,
                    filename='{epoch:02d}-{val_loss:.2f}',
                    save_top_k=3,
                    mode='min',
                    save_last=True
    )
    VAE=CVAE(learning_rate=0.0005,
             resolutoion=32,
             in_ch=3,
             ch=8,
             ch_mult=ch_mult,
             numhead=8,
             loss=loss,
             guide=False
             )
    trainer=L.Trainer(max_epochs=100,benchmark=True,check_val_every_n_epoch=5,logger=ts_logger,callbacks=[img_logger,lr_callback,model_ckpt_callback])
    train_loader,val_loader=get_data(batch_size=64)
    #train_loader,val_loader=get_cifar10_data(batch_size=128)
    trainer.fit(VAE,train_dataloaders=train_loader,val_dataloaders=val_loader)
