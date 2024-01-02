from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
import numpy as np
def color_fidelity(gt, rec):
    """
    Calculate the color fidelity for a batch of images.
    This function assumes that the images are in the format [B, C, H, W].
    """
    # Initialize the color fidelity list
    color_fid_values = []

    # Iterate over each image in the batch
    for i in range(gt.shape[0]):
        # Extract the individual images
        img1 = gt[i]
        img2 = rec[i]

        # Calculate the mean squared error for each color channel
        mse_r = np.mean((img1[0, :, :] - img2[0, :, :]) ** 2)
        mse_g = np.mean((img1[1, :, :] - img2[1, :, :]) ** 2)
        mse_b = np.mean((img1[2, :, :] - img2[2, :, :]) ** 2)

        # Combine the MSE of each channel to get the overall color fidelity for the image
        color_fid = np.sqrt((mse_r + mse_g + mse_b) / 3)
        color_fid_values.append(color_fid)

    # Return the average color fidelity for the batch
    return np.mean(color_fid_values)

def eval_batch(gt, rec):
    gt = gt.detach().cpu().numpy()
    rec = rec.detach().cpu().numpy()
    ssim_values = []
    psnr_values = []
    nrmse_values = []

    for i in range(gt.shape[0]):  # 遍历批次中的每一对图像
        img1 = gt[i]  # 从 (B, C, H, W) 转换为 (C, H, W)
        img2 = rec[i]  # 同上

        # 计算 SSIM
        ssim_val = ssim(img1.transpose(1, 2, 0), img2.transpose(1, 2, 0), 
                        data_range=img2.max() - img2.min(), multichannel=True,channel_axis=-1)
        ssim_values.append(ssim_val)

        # 计算 PSNR
        psnr_val = psnr(img1, img2, data_range=img2.max() - img2.min())
        psnr_values.append(psnr_val)

        # 计算 NRMSE
        nrmse_val = nrmse(img1, img2)
        nrmse_values.append(nrmse_val)

    # 计算色彩保真度
    color_fid_value = color_fidelity(gt, rec)

    ssim_value = sum(ssim_values) / len(ssim_values)
    psnr_value = sum(psnr_values) / len(psnr_values)
    nrmse_value = sum(nrmse_values) / len(nrmse_values)

    return dict(ssim=ssim_value, psnr=psnr_value, nrmse=nrmse_value, color_fid=color_fid_value)