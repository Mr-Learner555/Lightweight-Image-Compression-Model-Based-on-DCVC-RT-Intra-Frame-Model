import sys
import os

from src.models.img_prune import DMCI as DMCI_prune
from src.models.image_model4train2 import DMCI
from torchinfo import summary
from thop import profile
import torch
from src.utils.common import get_state_dict
from torch import nn
import random
from src.datasets.image_dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.transforms import rgb2ycbcr

# 函数
def calc_channel_importance(layer: nn.Conv2d) -> torch.Tensor:
    """计算Conv2d层每个输出通道的L1范数（重要性排序）"""
    weight = layer.weight.data  # [out_ch, in_ch, kH, kW]
    channel_l1 = torch.norm(weight, p=1, dim=(1,2,3))  # 按输出通道聚合L1
    return channel_l1

# 初始化模型
ckp_dir = "./experiments/cvpr2025_image.pth.tar"
# ckp_prune = "./experiments/prune_dmci.pth.tar"
model = DMCI(N=256, z_channel=128)
# model = DMCI_prune(N=256, z_channel=128)
pretained_model=get_state_dict(ckp_dir)
# pretained_model = torch.load(ckp_prune, map_location="cuda")
model.load_state_dict(pretained_model)
model.eval()

# 虚拟输入（匹配图像压缩模型的输入尺寸）
# dummy_input = torch.randn(1, 3, 768, 512)  # batch=1, 3通道, 256×256

# 1. 参数量/结构分析
# summary(model, input_data=(dummy_input, torch.tensor([4])))  # qp=4

# 2. FLOPs/参数量计算
qp = random.randint(0, 63)
flops = 0
params = 0
batch_size_test = 1
test_trans = transforms.Compose([transforms.ToTensor()])
test_set = ImageFolder("./test_image/kodak", transform=test_trans, split="")
test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=False, num_workers=2)
for img_tensor in tqdm(test_loader):
    img_tensor_ycbcr = rgb2ycbcr(img_tensor)
    flops_temp, params_temp = profile(model, inputs=(img_tensor_ycbcr, torch.tensor([qp])))
    flops += flops_temp
    params += params_temp
flops /= len(test_loader)
params /= len(test_loader)
print(f"原始FLOPs: {flops/1e9:.2f} G | 原始参数量: {params/1e6:.2f} M")

# 3. 精度基线（图像压缩用PSNR/SSIM）
# 需基于验证集评估原始模型的PSNR/SSIM，作为剪枝后恢复的目标


# 遍历模型，计算所有DepthConvBlock的1×1 Conv通道重要性
importance_dict = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) and module.kernel_size == (1,1):
        if "DepthConvBlock" in name:  # 仅分析深度卷积块内的1×1 Conv
            importance = calc_channel_importance(module)
            importance_dict[name] = importance