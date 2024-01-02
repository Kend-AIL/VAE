import torch.nn as nn
import torch
import lpips
class LPIPS_MAE_klloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual_loss = lpips.LPIPS(net='alex')
        
    def forward(self,gt,rec,pos,global_step,split):
        p_weight=1
        if global_step<1001:
            kl_weight=0.0000075
        else:
            kl_weight=0.00075
        rec_weight=1
        p_loss = self.perceptual_loss(rec.contiguous(), gt.contiguous())
        p_loss=p_loss.squeeze().mean()
        rec_loss=torch.abs(rec.contiguous()- gt.contiguous())
        rec_loss=rec_loss.mean()
        kl_loss=pos.kl().mean()
        all_loss=p_weight*p_loss+rec_weight*rec_loss+kl_weight*kl_loss

        log = {"{}/all_loss".format(split): all_loss.clone().detach(), 
               "{}/rec_loss".format(split): rec_loss.clone().detach(), 
                "{}/p_loss".format(split): p_loss.clone().detach(),
                "{}/kl_loss".format(split): kl_loss.clone().detach(), 
                   }
        return all_loss,log