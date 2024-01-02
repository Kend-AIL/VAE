import torch
import torch.nn as nn
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from modules.crossattn import MultiHeadCrossAttention
from modules.automodule import SimpleEncoder,SimpleDecoder
from modules.gauss import DiagonalGaussianDistribution
from modules.evaluation import eval_batch
class CVAE(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 resolutoion,
                 in_ch,
                 ch,
                 ch_mult,
                 numhead,
                 loss,
                 guide=False
                 ):
        super().__init__()
        self.lr=learning_rate
        self.encoder=SimpleEncoder(in_channels=in_ch,resolution=resolutoion,ch=ch,ch_mult=ch_mult)
        self.decoder=SimpleDecoder(in_channels=in_ch,resolution=resolutoion,ch=ch,ch_mult=ch_mult)
        hidden_dim=ch*ch_mult[-1]
        self.encode_conv= torch.nn.Conv2d(hidden_dim, 2*hidden_dim, 1)
        self.loss=loss
        self.guide=guide
        if self.guide:
            self.cross=MultiHeadCrossAttention(hidden_dim,numhead)
            self.embedding = nn.Embedding(100, 128)
            
    def encode(self,x):
        z=self.encoder(x)
        z=self.encode_conv(z)
        posterior = DiagonalGaussianDistribution(z)
        return posterior
    def decode(self,z,condition=None):
        if self.guide and condition is not None:
            guide=self.embedding(condition)
            guide_z=self.cross(guide,z)
            out=self.decoder(guide_z)
        
            return out
        elif self.guide==False:
            out=self.decoder(z)
            return out
        else:
            raise RuntimeError('No guide input!')
    def forward(self,x,condition=None):
            posterior=self.encode(x)
            z=posterior.sample()
            out=self.decode(z,condition)
            return out,posterior
    def configure_optimizers(self):
        lr = self.lr
        if self.guide:
            opt = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.encode_conv.parameters())+
                                  list(self.cross.parameters())+
                                  list(self.embedding.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        else:
            opt = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.encode_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt
    @torch.no_grad()
    def log_images(self, batch,**kwargs):
        log = dict()
        if self.guide:
            x= batch['image']
            label=batch['label']
            x= x.to(self.device)
            label=label.to(self.device)
            x_rec,pos=self(x,label)
            log["reconstructions"] = x_rec
            log["inputs"] = x
            
            log["samples"] = self.decode(torch.normal(torch.zeros(pos.sample().shape).to(self.device),torch.ones(pos.sample().shape).to(self.device)),label)
            
        else:
            x= batch['image']
            x= x.to(self.device)
            x_rec,pos=self(x)
            log["reconstructions"] = x_rec
            log["inputs"] = x
            log["samples"] =self.decode(torch.normal(torch.zeros(pos.sample().shape).to(self.device),torch.ones(pos.sample().shape).to(self.device)))
        return log
    def training_step(self, batch, batch_idx):
        inputs=batch['image']
        label = batch['label']
        if self.guide:
            rec,pos = self(inputs,label)
        else:
            rec,pos=self(inputs)
        
        loss,log_dict=self.loss(inputs,rec,pos,self.global_step,split="train")
        self.log('train/loss',loss,on_step=True,on_epoch=True,prog_bar=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        inputs=batch['image']
        label = batch['label']
        if self.guide:
            rec,pos = self(inputs,label)
        else:
            rec,pos=self(inputs)
        loss,log_dict=self.loss(inputs,rec,pos,self.global_step,split="val")
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log('val_loss', loss,logger=True,on_step=False, on_epoch=True)
        eval_dict=eval_batch(rec,inputs)
        self.log_dict(eval_dict,on_epoch=True)
        return self.log_dict
    