import torch
from collections import OrderedDict
from src.models.img_prune import DMCI   # 请替换为实际导入
from src.utils.common import get_state_dict

def load_state_dict_from_checkpoint(path):
    """兼容直接保存的 state_dict 或包含 'state_dict' 的字典"""
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint

def merge_dmci_weights(
    path_enc,          # checkpoint1: 包含 enc + q_scale_enc
    path_dec,          # checkpoint2: 包含 dec + q_scale_dec
    path_rest,         # checkpoint3: 其余所有层（与 DMCI 一致）
    model_class=None,  # DMCI 类，用于校验参数名（可选）
):
    # 1. 加载三个权重
    state1 = load_state_dict_from_checkpoint(path_enc)
    state2 = load_state_dict_from_checkpoint(path_dec)
    state3 = get_state_dict(path_rest)


    # 2. 从 checkpoint1 提取 enc + q_scale_enc
    enc_state = OrderedDict()
    for k, v in state1.items():
        if k.startswith('enc.') or k == 'q_scale_enc':
            enc_state[k] = v

    # 3. 从 checkpoint2 提取 dec + q_scale_dec
    dec_state = OrderedDict()
    for k, v in state2.items():
        if k.startswith('dec.') or k == 'q_scale_dec':
            dec_state[k] = v

    # 4. 从 checkpoint3 提取其余参数（排除了 enc/dec/q_scale_*）
    rest_state = OrderedDict()
    excluded_prefixes = ('enc.', 'dec.', 'q_scale_enc', 'q_scale_dec')
    for k, v in state3.items():
        if not any(k.startswith(pre) or k == pre for pre in excluded_prefixes):
            rest_state[k] = v

    # 5. 合并（顺序：先放 enc 再放 dec 再放其余，顺序无关紧要）
    merged = OrderedDict()
    merged.update(enc_state)
    merged.update(dec_state)
    merged.update(rest_state)

    # 6. 可选：校验是否与模型参数名完全匹配
    if model_class is not None:
        dummy_model = model_class()
        model_keys = set(dummy_model.state_dict().keys())
        merged_keys = set(merged.keys())

        missing = model_keys - merged_keys
        unexpected = merged_keys - model_keys
        if missing:
            print(f"警告: 合并后缺少下列参数: {missing}")
        if unexpected:
            print(f"警告: 合并后包含多余参数: {unexpected}")
        if not missing and not unexpected:
            print("✓ 合并后的 state_dict 与 DMCI 模型完全匹配")

    return merged


# ========== 使用示例 ==========
if __name__ == '__main__':
    # 修改为实际路径
    ckpt1_path = './experiments/B_Ep150-5-6-10.pth.tar' # enc
    ckpt2_path = './experiments/Ep150-5-2-10.pth.tar' # dec
    ckpt3_path = './experiments/cvpr2025_image.pth.tar' # rest

    merged_state_dict = merge_dmci_weights(ckpt1_path, ckpt2_path, ckpt3_path)

    # 保存合并后的权重
    torch.save(merged_state_dict, './experiments/prune_dmci_B.pth.tar')

    # 或者直接加载到模型
    model = DMCI()
    model.load_state_dict(merged_state_dict)
    print("模型权重加载成功")