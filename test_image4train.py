import torch
import time
from PIL import Image
from torchvision import transforms
from src.datasets.image_dataset import ImageFolder
from src.models.img_prune import DMCI as DMCI_pruned
from src.models.image_model4train2 import DMCI
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb
from src.utils.metrics import calc_msssim_rgb, calc_psnr
from src.utils.common import get_state_dict
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import json
import os


def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom


def replicate_pad(x, pad_b, pad_r):
        if pad_b == 0 and pad_r == 0:
            return x
        else:
            return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")

def pad_for_x(x):
        _, _, H, W = x.size()
        padding_r, padding_b = get_padding_size(H, W, 64)
        x_pad = replicate_pad(x, padding_b, padding_r)
        return x_pad

def compress_one_image_forward(model, x, qp):
    torch.cuda.synchronize()
    start_time = time.time()
    N, _, H, W = x.shape
    with torch.no_grad():
        out = model(x, qp)
    torch.cuda.synchronize()
    end_time = time.time()
    bpp = out["bits"] / (N * H * W)
    enc_time = end_time - start_time
    return out["x_hat"], bpp, enc_time


def compress_one_image(model, x, qp):
    torch.cuda.synchronize()
    start_time = time.time()
    N, _, H, W = x.shape
    with torch.no_grad():
        out = model.compress(x, qp)
    torch.cuda.synchronize()
    end_time = time.time()
    bit_stream = out["bit_stream"]

    bpp = len(bit_stream) * 8 / (N * H * W)
    enc_time = end_time - start_time
    return bit_stream, bpp, enc_time


def decompress_one_image(model, bit_stream, sps):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.decompress(bit_stream, sps)
    torch.cuda.synchronize()
    end_time = time.time()
    dec_time = end_time - start_time
    x_hat = out["x_hat"]
    return x_hat, dec_time


def testing_forward(i_frame_model, dataset, device, qpl):
    i_frame_model.eval()
    recon=None
    count=len(dataset)
    summary_results=[]
    with torch.no_grad():
        for q in qpl:
            avgbpp=0
            avgpsnr=0
            avgmsssim=0
            avg_forward_time=0
            for img_tensor in tqdm(dataset):
                img_tensor_ycbcr = rgb2ycbcr(img_tensor)
                if device.type == 'cuda':
                    img_tensor_ycbcr = img_tensor_ycbcr.to(device=device, non_blocking=True)
                x_hat, bpp, forward_time = compress_one_image_forward(i_frame_model, img_tensor_ycbcr, q)
                avgbpp+=bpp.item()
                avg_forward_time += forward_time
                recon_rgb=ycbcr2rgb(x_hat, clamp=True)
                recon_rgb=recon_rgb*255
                recon_rgb=recon_rgb.squeeze(0).cpu().numpy()
                rgb=torch.clamp(img_tensor*255, 0, 255).squeeze(0).cpu().numpy()
                psnr=calc_psnr(rgb, recon_rgb)
                msssim=calc_msssim_rgb(rgb, recon_rgb)
                avgpsnr+=psnr
                avgmsssim+=msssim
            avgbpp=avgbpp/count
            avgpsnr=avgpsnr/count
            avgmsssim=avgmsssim/count
            avg_forward_time=avg_forward_time/count
            # 就这样写保存的列表吧，简单些
            summary_stats = {'qp': q,
                             'Average bpp': avgbpp,
                             'Average psnr': avgpsnr,
                             'Average msssim': avgmsssim,
                             'forward latency': avg_forward_time,
                             'Count': count
                            }
            summary_results.append(summary_stats)
    return summary_results


def testing(i_frame_model, dataset, device, qpl):
    i_frame_model.eval()
    recon=None
    count=len(dataset)
    summary_results=[]
    with torch.no_grad():
        for q in qpl:
            avgbpp=0
            avgpsnr=0
            avgmsssim=0
            avg_encode_time=0
            avg_decode_time=0
            for img_tensor in tqdm(dataset):
                img_tensor = img_tensor.half()
                img_tensor_ycbcr = rgb2ycbcr(img_tensor)
                if device.type == 'cuda':
                    img_tensor_ycbcr = img_tensor_ycbcr.to(device=device, non_blocking=True)
                [original_h, original_w] = img_tensor_ycbcr.shape[2:]
                img_padded = pad_for_x(img_tensor_ycbcr)
                bit_stream, bpp, enc_time = compress_one_image(i_frame_model, img_padded, q)
                [curr_h, curr_w] = img_padded.shape[2:]

                sps = {
                    'sps_id': -1,
                    'height': curr_h,
                    'width': curr_w,
                    'ec_part': 0,
                    'use_ada_i': 0,
                    'qp': q,
                }

                x_hat, dec_time = decompress_one_image(i_frame_model, bit_stream, sps)
                recon = x_hat[:,:,:original_h,:original_w]
                avgbpp+=bpp
                avg_encode_time += enc_time
                avg_decode_time += dec_time
                recon_rgb=ycbcr2rgb(recon, clamp=True)
                recon_rgb=recon_rgb*255
                recon_rgb=recon_rgb.squeeze(0).cpu().numpy()
                rgb=torch.clamp(img_tensor*255, 0, 255).squeeze(0).cpu().numpy()
                psnr=calc_psnr(rgb, recon_rgb)
                msssim=calc_msssim_rgb(rgb, recon_rgb)
                avgpsnr+=psnr
                avgmsssim+=msssim
            avgbpp=avgbpp/count
            avgpsnr=avgpsnr/count
            avgmsssim=avgmsssim/count
            avg_encode_time=avg_encode_time/count
            avg_decode_time=avg_decode_time/count
            # 就这样写保存的列表吧，简单些
            summary_stats = {'qp': q,
                             'Average bpp': avgbpp,
                             'Average psnr': avgpsnr,
                             'Average msssim': avgmsssim,
                             'Encoding latency': avg_encode_time,
                             'Decoding latency': avg_decode_time,
                             'Count': count
                            }
            summary_results.append(summary_stats)
    return summary_results

if __name__ == '__main__':
    torch.manual_seed(42) # 这里我和common.py中的set_torch_env()一致
    np.random.seed(seed=42) # 依旧和set_torch_env()对齐
    random.seed(42) # 打乱图片路径
    batch_size_test = 1
    qpl = [i for i in range(0, 64)]
    ckp_dir="./checkpoints/prune_dmci.pth.tar"
    summaries=[]
    device = torch.device("cuda")
    test_max_imgs = 24
    # 测试集batch不打乱，用原来的batch
    test_trans = transforms.Compose([transforms.ToTensor()])

    test_set = ImageFolder("./test_image/kodak", transform=test_trans, split="")

    # 单卡 DataLoader
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    
    inet= DMCI_pruned()
    # pretained_model=get_state_dict(ckp_dir)
    # inet.load_state_dict(pretained_model)

    pretained_model = torch.load(ckp_dir, map_location=device)
    inet.load_state_dict(pretained_model)

    # pretained_model = torch.load(ckp_dir, map_location=device)
    # inet.load_state_dict(pretained_model["state_dict"])

    inet = inet.to(device)
    inet = inet.half() # 模型也使用半精度
    inet.update() # 必须要调用一下update()

    results = testing(inet, test_loader, device, qpl)

    test_log_path = os.path.join("./experiments", "real_prune_kodak_half.json")
    with open(test_log_path, 'w') as f:
        json.dump(results, f, indent=4)


    