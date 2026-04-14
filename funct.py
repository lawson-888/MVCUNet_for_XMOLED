from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def calculate_psnr_or_ssim(label_img, netout_img, data_range, goal, incre_degree):
    img1 = torch.permute(torch.squeeze(label_img, dim=0), [1, 2, 0]).detach().cpu().numpy() / incre_degree
    img2 = torch.permute(torch.squeeze(netout_img, dim=0), [1, 2, 0]).detach().cpu().numpy() / incre_degree
    if goal == 1:
        psnr = compare_psnr(img1, img2, data_range=data_range/incre_degree)
        return psnr
    else:
        ssim = compare_ssim(img1, img2, data_range=data_range/incre_degree, channel_axis=2)
        return ssim


def figure_save(label_image_save_path, test_image_save_path, label_img, testout_img, scale_degree):
    plt.figure()
    plt.imshow(torch.squeeze(label_img.detach().cpu() / scale_degree), cmap='jet', vmin=0, vmax=0.2)
    plt.colorbar()
    plt.savefig(label_image_save_path)
    plt.close()

    plt.figure()
    plt.imshow(torch.squeeze(testout_img.detach().cpu() / scale_degree), cmap='jet', vmin=0, vmax=0.2)
    plt.colorbar()
    plt.savefig(test_image_save_path)
    plt.close()
