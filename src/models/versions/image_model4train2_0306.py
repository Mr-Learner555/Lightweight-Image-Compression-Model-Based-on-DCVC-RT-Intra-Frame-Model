# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F


from compressai.ops import quantize_ste as ste_round
from src.layers.layers4train import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

# 通道数不一样了，这一点在DCVC-RT训练时，是否需要修改，怎么确定合适的值？
# g_ch_src = 8 * 8
# g_ch_enc_dec = 128

g_ch_src = 3 * 8 * 8
g_ch_enc_dec = 368

class IntraEncoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.enc_1 = DepthConvBlock(g_ch_src, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):

        out = F.pixel_unshuffle(x, 8)

        return self.forward_torch(out, quant_step)


    def forward_torch(self, out, quant_step):
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(N, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.dec_2 = DepthConvBlock(g_ch_enc_dec, g_ch_src)

    def forward(self, x, quant_step):

        return self.forward_torch(x, quant_step)


    def forward_torch(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out


def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_right = new_w - width
    padding_bottom = new_h - height
    return padding_right, padding_bottom

def pad_for_y(y):
    _, _, H, W = y.size()
    padding_r, padding_b = get_padding_size(H, W, 4)
    y_pad = F.pad(y, (0, padding_r, 0, padding_b), mode="replicate")
    
    return y_pad
    
class DMCI(nn.Module):
    def __init__(self, N=256, z_channel=128):
        super().__init__()

        self.enc = IntraEncoder(N)
        
        self.entropy_bottleneck = EntropyBottleneck(z_channel)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
        )

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel),
            ResidualBlockUpsample(z_channel, z_channel),
            DepthConvBlock(z_channel, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2 + 2, 1),
        )

        self.y_spatial_prior_reduction = nn.Conv2d(N * 2 + 2, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_4 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2, 1),
        )

        self.dec = IntraDecoder(N)

        self.q_scale_enc = nn.Parameter(torch.ones((64, g_ch_enc_dec, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((64, g_ch_enc_dec, 1, 1)))
        self.masks = {}

    def process_with_mask(self, y, scales, means, mask, force_zero_thres = None):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = torch.round(y_res)
        if force_zero_thres is not None:
            cond = scales_hat > force_zero_thres
            y_q = y_q * cond
        y_q = torch.clamp(y_q, -128., 127.)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    def separate_prior(self, params, is_video=False):
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = torch.clamp_min(quant_step, 0.5)
            q_enc = 1. / quant_step
            q_dec = quant_step
        else:
            q = params[:, :2, :, :]
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

    @staticmethod
    def get_one_mask(micro_mask, height, width, dtype, device):
        mask = torch.tensor(micro_mask, dtype=dtype, device=device)
        mask = mask.repeat((height + 1) // 2, (width + 1) // 2)
        mask = mask[:height, :width]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)
        return mask
    
    def get_mask_4x(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_4x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0 = self.get_one_mask(((1, 0), (0, 0)), height, width, dtype, device)
                m1 = self.get_one_mask(((0, 1), (0, 0)), height, width, dtype, device)
                m2 = self.get_one_mask(((0, 0), (1, 0)), height, width, dtype, device)
                m3 = self.get_one_mask(((0, 0), (0, 1)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]
    
    # 前向传播函数中没用到4x掩码压缩方法，可能是不可微，关于掩码难道没有可微的解决方法吗？
    def forward(self, x, qp):

        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        y = self.enc(x, curr_q_enc)
        hyper_inp = pad_for_y(y)
        z = self.hyper_enc(hyper_inp)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_hat = torch.clamp(torch.round(z), -128., 127.)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        common_params = params[:, :, :yH, :yW].contiguous()

        q_enc, q_dec, scales, means = self.separate_prior(common_params, False)
        common_params = self.y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        y = y * q_enc

        _, _, y_hat_0, _ = self.process_with_mask(y, scales, means, mask_0, None)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.y_spatial_prior(self.y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        _, _, y_hat_1, _ = self.process_with_mask(y, scales, means, mask_1, None)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.y_spatial_prior(self.y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        _, _, y_hat_2, _ = self.process_with_mask(y, scales, means, mask_2, None)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.y_spatial_prior(self.y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        _, _, y_hat_3, _ = self.process_with_mask(y, scales, means, mask_3, None)

        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec

        params = torch.cat((y_hat, common_params), dim=1)
        scales, means = self.y_spatial_prior(self.y_spatial_prior_adaptor_4(params)).chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales, means)
        # y_hat, y_likelihoods = self.gaussian_conditional(y, params)
        
        x_hat = self.dec(y_hat, curr_q_dec)#.clamp_(0, 1)
        
        return  x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods

    def compress(self, x, qp):

        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]
        
        y = self.enc(x, curr_q_enc)
        hyper_inp = pad_for_y(y)
        z = self.hyper_enc(hyper_inp)

        z_strings = self.entropy_bottleneck.compress(z)

        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        indexes = self.gaussian_conditional.build_indexes(params)
        y_strings = self.gaussian_conditional.compress(y, indexes)

        return y_strings, z_strings, z.size()[-2:], y.shape

    def decompress(self, y_strings, z_strings, shapez, shapey, qp):

        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        z_hat = self.entropy_bottleneck.decompress(z_strings, shapez)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = shapey
        params = params[:, :, :yH, :yW].contiguous()
        
        indexes = self.gaussian_conditional.build_indexes(params)
        
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, z_hat.dtype)
        
        x_hat = self.dec(y_hat, curr_q_dec)#.clamp_(0, 1)
        
        return x_hat
