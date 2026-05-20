import sys
import os

import argparse
import math
import random
from tqdm import tqdm
import json

import torch
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.image_model4train2 import DMCI as DMCI
from src.models.img_prune import DMCI as DMCI_pruned
from src.datasets.image_dataset import ImageFolder
from src.utils.common import get_state_dict
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb
from src.utils.metrics import calc_psnr, calc_msssim_rgb

# -----------------------------------------------------------
# 1. 损失函数 (完全保留原逻辑)
# -----------------------------------------------------------
class RateDistortionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}
        bpp_total = 0
        for name, likelihoods in output["likelihoods"].items():
            bpp = torch.sum(-torch.log2(torch.clamp(likelihoods, 1e-9, 1.0))) / num_pixels
            out[f"bpp_{name}"] = bpp
            bpp_total += bpp
        
        out["bpp_loss"] = bpp_total
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = lmbda * (255 ** 2) * out["mse_loss"] + out["bpp_loss"]
        return out



def load_pretrained_for_slim_dmci(checkpoint_path, new_model, old_g_ch=368, new_g_ch=300):
    """
    从预训练 checkpoint 加载权重到缩小了 g_ch_enc_dec 的 DMCI 模型。
    
    参数：
        checkpoint_path: 预训练 .pth.tar 文件路径
        new_model: 已实例化的窄 DMCI 模型（例如 DMCI(N=256, z_channel=128, g_ch_enc_dec=256)）
        old_g_ch: 原始模型的 g_ch_enc_dec (默认 368)
        new_g_ch: 新模型的 g_ch_enc_dec (默认 256)
    返回：
        new_model: 加载好权重的窄模型
    """
    # # 加载 checkpoint
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # if 'state_dict' in checkpoint:
    #     pretrained_dict = checkpoint['state_dict']
    # elif 'model' in checkpoint:
    #     pretrained_dict = checkpoint['model']
    # else:
    #     pretrained_dict = checkpoint

    pretrained_dict=get_state_dict(checkpoint_path)

    # 获取新模型的 state_dict（随机初始化）
    new_state = new_model.state_dict()

    # 逐层迁移权重
    for name, new_param in new_state.items():
        if name in pretrained_dict:
            old_param = pretrained_dict[name]
            if old_param.shape == new_param.shape:
                new_state[name] = old_param
            else:
                # 形状不一致时，按新张量的每个维度裁剪（保留前 min 个元素）
                # 适用于输出/输入通道减少的情况
                slices = tuple(
                    slice(0, min(old, new)) for old, new in zip(old_param.shape, new_param.shape)
                )
                trimmed = old_param[slices]
                # 确保裁剪后形状完全匹配
                if trimmed.shape == new_param.shape:
                    new_state[name] = trimmed
                else:
                    # 极少情况（例如权重转置差异）才需要填充，这里假设仅缩小，不会走到
                    raise RuntimeError(f"Cannot match shape for {name}: {old_param.shape} -> {new_param.shape}")
        else:
            print(f"Warning: {name} not found in pretrained checkpoint, using random init.")

    new_model.load_state_dict(new_state)
    print(f"Successfully transferred weights with g_ch_enc_dec {old_g_ch} -> {new_g_ch}")
    return new_model


# -----------------------------------------------------------
# 2. 训练函数 (移除多卡逻辑)
# -----------------------------------------------------------
def train_one_epoch(model, teacher, criterion, train_dataloader, optimizer, epoch, device):
    model.train()
    teacher.eval()
    
    train_log=[]

    pbar = tqdm(train_dataloader, desc="Training")
    for i, d in enumerate(pbar):
        d = d.to(device)
        d_ycbcr = rgb2ycbcr(d)
        optimizer.zero_grad()

        qp_index = random.randint(0, 63)

        # forward ing
        curr_q_enc = model.q_scale_enc[qp_index:qp_index+1, :, :, :]
        curr_q_enc_teach = teacher.q_scale_enc[qp_index:qp_index+1, :, :, :]

        y = model.enc(d_ycbcr, curr_q_enc)
        y_teach = teacher.enc(d_ycbcr, curr_q_enc_teach)
        
        y_mse = criterion.mse(y, y_teach)

        # 暂时不动，先保持这个损失函数
        if y_mse >= 0.1:
            loss = y_mse * 30
        elif y_mse >= 0.01 and y_mse < 0.1:
            loss = y_mse * 50
        elif y_mse >= 0.001 and y_mse < 0.01:
            loss = y_mse * 50
        else:
            loss = y_mse * 100

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "y_mse": f"{y_mse.item():.4f}", "lr": f"{optimizer.state_dict()['param_groups'][0]['lr']}"})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
            
        if i % 50 == 0:
            train_stats = {
                'Process': f"Epoch {epoch+1} - Iteration {i}", 
                'qp': qp_index,
                'loss': loss.item(),
                "y_mse": y_mse.item(),
                "lr": f"{optimizer.state_dict()['param_groups'][0]['lr']}"
            }
            train_log.append(train_stats)
    return train_log

# -----------------------------------------------------------
# 3. 测试函数 (移除多卡聚合逻辑)
# -----------------------------------------------------------
def test_epoch(epoch, test_dataloader, model, model2, teacher, device):
    model.eval()
    teacher.eval()

    qpl = [i for i in range(0, 64, 8)]
    qpl.append(63)
    test_log = []
    with torch.no_grad():
        for test_qp in qpl:
            psnr_sum = 0.0
            msssim_sum = 0.0
            bpp_sum = 0.0
            count = 0
            for d in test_dataloader:
                d = d.to(device)
                d_ycbcr = rgb2ycbcr(d)
                qp_tensor = torch.tensor([test_qp], device=device)
                
                curr_q_enc = model.q_scale_enc[test_qp:test_qp+1, :, :, :]
                curr_q_dec = model2.q_scale_dec[test_qp:test_qp+1, :, :, :]

                y = model.enc(d_ycbcr,curr_q_enc)
                y_pad = teacher.pad_for_y(y)
                z = teacher.hyper_enc(y_pad)
                z_hat = torch.clamp(teacher.ste_round(z), -128., 127.)
                params = teacher.hyper_dec(z_hat)
                params = teacher.y_prior_fusion(params) 
                _, _, yH, yW = y.shape #获取特征图高度
                params = params[:, :, :yH, :yW].contiguous() #裁剪尺寸对齐
                
                y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat = \
                    teacher.compress_prior_4x(
                        y, params, teacher.y_spatial_prior_reduction,
                        teacher.y_spatial_prior_adaptor_1, teacher.y_spatial_prior_adaptor_2,
                        teacher.y_spatial_prior_adaptor_3, teacher.y_spatial_prior, True)
                # dec from student
                x_hat = model2.dec(y_hat, curr_q_dec)
            
                x_hat_rgb = ycbcr2rgb(x_hat, clamp=True)
                
                # 比特估计：分别对四个分区的合并符号计算高斯比特
                bits_y = (teacher.get_y_gaussian_bits(y_q_w_0, s_w_0) +
                        teacher.get_y_gaussian_bits(y_q_w_1, s_w_1) +
                        teacher.get_y_gaussian_bits(y_q_w_2, s_w_2) +
                        teacher.get_y_gaussian_bits(y_q_w_3, s_w_3))
                bits_z = teacher.get_z_bits(z, teacher.bit_estimator_z, qp_tensor)

                bits = bits_y.sum() + bits_z.sum()
                N, _, H, W = d.size()
                num_pixels = N * H * W
                bpp = bits / num_pixels

                x_hat_rgb=x_hat_rgb*255
                x_hat_rgb=x_hat_rgb.squeeze(0).cpu().numpy()
                d=torch.clamp(d*255, 0, 255).squeeze(0).cpu().numpy()
                psnr=calc_psnr(d, x_hat_rgb)
                msssim=calc_msssim_rgb(d, x_hat_rgb)

                psnr_sum += psnr
                msssim_sum += msssim
                bpp_sum += bpp.item()
                count += 1

            avg_psnr = psnr_sum / count
            avg_msssim = msssim_sum / count
            avg_bpp = bpp_sum / count
            test_stats = {
                'Process': f"Epoch {epoch+1}", 
                'qp': test_qp,
                'Average bpp': avg_bpp,
                'Average psnr': avg_psnr,
                'Average msssim': avg_msssim,
            }
            test_log.append(test_stats)
    return test_log

# -----------------------------------------------------------
# 4. 单卡主函数 (核心修改)
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    # 默认训练20轮
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--stu_ckp_dir', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    # 单卡设备
    device = "cuda" 
    # 数据预处理
    train_trans = transforms.Compose([transforms.RandomCrop((256, 256)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])
    test_trans = transforms.Compose([transforms.ToTensor()])

    train_set = ImageFolder(args.train_dataset, transform=train_trans, split="")
    test_set = ImageFolder(args.test_dataset, transform=test_trans, split="")

    # 单卡 DataLoader，直接 shuffle
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    train_log = []
    test_log = []
    # 模型初始化
    net = DMCI_pruned(N=256, z_channel=128).to(device)
    net2 = DMCI_pruned(N=256, z_channel=128).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.25)
    criterion = RateDistortionLoss()
    teacher = DMCI(N=256, z_channel=128).to(device)

    # 加载断点
    start_epoch = 0
    checkpoint_path = os.path.join(args.save_dir, "cvpr2025_image.pth.tar")
    print(f"加载 checkpoint: {checkpoint_path}")
    pretained_model=get_state_dict(checkpoint_path)
    teacher.load_state_dict(pretained_model)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    net2_ckp_path = os.path.join(args.save_dir, "Ep150-5-2-10.pth.tar")
    print(f"load dec checkpoint: {net2_ckp_path}")
    net2_pretained_model=torch.load(net2_ckp_path, map_location=device)
    net2.load_state_dict(net2_pretained_model["state_dict"])
    for param in net2.parameters():
        param.requires_grad = False
    net2.eval()

    if args.stu_ckp_dir == "nope":
        net = load_pretrained_for_slim_dmci(
        checkpoint_path=checkpoint_path,
        new_model=net,
        old_g_ch=368,
        new_g_ch=300
        )
        net = net.to(device)
    else:
        stu_ckp_path = os.path.join(args.save_dir, args.stu_ckp_dir)
        if os.path.exists(stu_ckp_path):
            print(f"Load student checkpoint: {stu_ckp_path}")
            stu_ckp = torch.load(stu_ckp_path, map_location=device)
            net.load_state_dict(stu_ckp['state_dict'])
            # optimizer.load_state_dict(stu_ckp['optimizer'])
            start_epoch = stu_ckp.get('epoch', 0)
    for param in net.parameters():
        param.requires_grad = False
    for param in net.enc.parameters():
        param.requires_grad = True
    net.q_scale_enc.requires_grad = True
    net.enc.train()

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        print(f"\n========== 开始训练 Epoch {epoch+1} ==========")
        train_info = train_one_epoch(net, teacher, criterion, train_loader, optimizer, epoch, device)
        train_log.extend(train_info)
        test_info = test_epoch(epoch, test_loader, net, net2, teacher, device)
        test_log.extend(test_info)
        lr_scheduler.step()
        if (epoch + 1) % args.epochs == 0:
            # 保存模型
            state = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.save_dir, f"Ep{epoch+1}-{time.localtime().tm_mon}-{time.localtime().tm_mday}-{time.localtime().tm_hour}.pth.tar"))

    train_log_path = os.path.join(args.save_dir, f"train_log-{time.localtime().tm_mon}-{time.localtime().tm_mday}-{time.localtime().tm_hour}.json")
    with open(train_log_path, 'w') as f:
        json.dump(train_log, f, indent=4)
    test_log_path = os.path.join(args.save_dir, f"test_log-{time.localtime().tm_mon}-{time.localtime().tm_mday}-{time.localtime().tm_hour}.json")
    with open(test_log_path, 'w') as f:
        json.dump(test_log, f, indent=4)
    print(f"{args.epochs}轮训练全部完成!")

if __name__ == "__main__":
    main()