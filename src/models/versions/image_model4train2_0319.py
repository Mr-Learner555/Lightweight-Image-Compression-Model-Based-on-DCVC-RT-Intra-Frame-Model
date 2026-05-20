# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F

import time
from compressai.ans import BufferedRansEncoder, RansDecoder
import torch.backends.cudnn as cudnn
from compressai.models import CompressionModel
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

    def forward_cuda(self, out, quant_step):
        out = self.enc_1(out, quant_step=quant_step)
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

    def forward_cuda(self, x, quant_step):
        out = self.dec_1[0](x)
        out = self.dec_1[1](out)
        out = self.dec_1[2](out)
        out = self.dec_1[3](out)
        out = self.dec_1[4](out)
        out = self.dec_1[5](out)
        out = self.dec_1[6](out)
        out = self.dec_1[7](out)
        out = self.dec_1[8](out)
        out = self.dec_1[9](out)
        out = self.dec_1[10](out)
        out = self.dec_1[11](out)
        out = self.dec_1[12](out, quant_step=quant_step)
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
    
class DMCI(CompressionModel):
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
        # self.y_spatial_prior_adaptor_4 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
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

    def process_with_mask(self, y, scales, means, mask, ste_round, force_zero_thres=None):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = ste_round(y_res)
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
    
    def squeeze_with_mask(self, latent, mask):
        latent_group_1, latent_group_2, latent_group_3, latent_group_4 = latent.chunk(4, 1)
        mask_group_1, mask_group_2, mask_group_3, mask_group_4 = mask.chunk(4, 1)
        latent_squeeze = (latent_group_1 * mask_group_1 + latent_group_2 * mask_group_2 +
                          latent_group_3 * mask_group_3 + latent_group_4 * mask_group_4)
        return latent_squeeze

    def unsqueeze_with_mask(self, latent_squeeze, mask):
        mask_group_1, mask_group_2, mask_group_3, mask_group_4 = mask.chunk(4, 1)
        latent = torch.cat((latent_squeeze * mask_group_1,
                            latent_squeeze * mask_group_2,
                            latent_squeeze * mask_group_3,
                            latent_squeeze * mask_group_4), dim=1)
        return latent

    def compress_group_with_mask(self, latent, scales, means, mask, symbols_list, indexes_list):
        latent_sqz = self.squeeze_with_mask(latent, mask)
        scales_sqz = self.squeeze_with_mask(scales, mask)
        means_sqz = self.squeeze_with_mask(means, mask)
        indexes = self.gaussian_conditional.build_indexes(scales_sqz)
        latent_sqz_hat = self.gaussian_conditional.quantize(latent_sqz, "symbols", means_sqz)
        symbols_list.extend(latent_sqz_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        latent_hat = self.unsqueeze_with_mask(latent_sqz_hat + means_sqz, mask)
        return latent_hat
    
    def get_downsampled_shape(self, height, width, p):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return int(new_h / p + 0.5), int(new_w / p + 0.5)
    
    def decompress_group_with_mask(self, scales, means, mask, decoder, cdf, cdf_lengths, offsets):
        scales_squeeze = self.squeeze_with_mask(scales, mask)
        means_squeeze = self.squeeze_with_mask(means, mask)
        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        latent_squeeze_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        latent_squeeze_hat = torch.Tensor(latent_squeeze_hat).reshape(scales_squeeze.shape).to(scales.device)
        latent_hat = self.unsqueeze_with_mask(latent_squeeze_hat + means_squeeze, mask)
        return latent_hat

    # 前向传播函数中没用到4x掩码压缩方法，可能是不可微，关于掩码难道没有可微的解决方法吗？
    def forward(self, x, qp):
        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        y = self.enc(x, curr_q_enc)
        # H_y, W_y = 16
        # hyper_inp = pad_for_y(y)
        z = self.hyper_enc(y)
        _, z_likelihoods = self.entropy_bottleneck(z, True)

        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        params_raw = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params_raw)
        _, _, yH, yW = y.shape
        common_params_all = params[:, :, :yH, :yW].contiguous()

        q_enc, q_dec, scales_0, means_0 = self.separate_prior(common_params_all, False)
        common_params = self.y_spatial_prior_reduction(common_params_all)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        y_scaled = y * q_enc

        _, _, y_hat_0, _ = self.process_with_mask(y_scaled, scales_0, means_0, mask_0, ste_round, None)

        y_hat_so_far = y_hat_0
        params_0 = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_1, means_1 = self.y_spatial_prior(self.y_spatial_prior_adaptor_1(params_0)).chunk(2, 1)
        _, _, y_hat_1, _ = self.process_with_mask(y_scaled, scales_1, means_1, mask_1, ste_round, None)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params_1 = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_2, means_2 = self.y_spatial_prior(self.y_spatial_prior_adaptor_2(params_1)).chunk(2, 1)
        _, _, y_hat_2, _ = self.process_with_mask(y_scaled, scales_2, means_2, mask_2, ste_round, None)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params_2 = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_3, means_3 = self.y_spatial_prior(self.y_spatial_prior_adaptor_3(params_2)).chunk(2, 1)
        _, _, y_hat_3, _ = self.process_with_mask(y_scaled, scales_3, means_3, mask_3, ste_round, None)

        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec

        scales_all = scales_0 * mask_0 + scales_1 * mask_1 + scales_2 * mask_2 + scales_3 * mask_3
        means_all = means_0 * mask_0 + means_1 * mask_1 + means_2 * mask_2 + means_3 * mask_3
        # _, y_likelihoods = self.gaussian_conditional(y_scaled, scales_all, means_all)
        _, y_likelihoods = self.gaussian_conditional(y_scaled, scales_all, means_all, True)

        # params_3 = torch.cat((y_hat, common_params), dim=1)
        # scales_4, means_4 = self.y_spatial_prior(self.y_spatial_prior_adaptor_4(params_3)).chunk(2, 1)
        # _, y_likelihoods = self.gaussian_conditional(y_scaled, scales_4, means_4)
        # y_hat, y_likelihoods = self.gaussian_conditional(y, params)
        
        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)
        
        return  x_hat, y_likelihoods, z_likelihoods

    def compress(self, x, qp):
        cudnn.deterministic = True
        device = x.device
        dtype = x.dtype
        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        _, _, x_height, x_width = x.shape
        y = self.enc(x, curr_q_enc)
        z = self.hyper_enc(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        params_raw = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params_raw)
        _, _, yH, yW = y.shape
        common_params_all = params[:, :, :yH, :yW].contiguous()

        q_enc, _, scales_0, means_0 = self.separate_prior(common_params_all, False)
        y_scaled = y * q_enc
        common_params = self.y_spatial_prior_reduction(common_params_all)

        B, C, H, W = y_scaled.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)
        y_hat_0 = self.compress_group_with_mask(y_scaled, scales_0, means_0, mask_0, symbols_list, indexes_list)
        y_hat_so_far = y_hat_0

        params_0 = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_1, means_1 = self.y_spatial_prior(self.y_spatial_prior_adaptor_1(params_0)).chunk(2, 1)
        y_hat_1 = self.compress_group_with_mask(y_scaled, scales_1, means_1, mask_1, symbols_list, indexes_list)


        y_hat_so_far = y_hat_so_far + y_hat_1
        params_1 = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_2, means_2 = self.y_spatial_prior(self.y_spatial_prior_adaptor_2(params_1)).chunk(2, 1)
        y_hat_2 = self.compress_group_with_mask(y_scaled, scales_2, means_2, mask_2, symbols_list, indexes_list)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params_2 = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_3, means_3 = self.y_spatial_prior(self.y_spatial_prior_adaptor_3(params_2)).chunk(2, 1)
        _ = self.compress_group_with_mask(y_scaled, scales_3, means_3, mask_3, symbols_list, indexes_list)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        cudnn.deterministic = False

        return {
            "strings": [y_strings, z_strings],
            "sps": [qp, x_height, x_width]
        }

    def decompress(self, strings, sps):
        cudnn.deterministic = True
        torch.cuda.synchronize()
        start_time = time.process_time()

        y_strings = strings[0][0]
        z_strings = strings[1]
        qp = sps[0]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]
        z_shape = self.get_downsampled_shape(sps[1], sps[2], 64)
        yH, yW = self.get_downsampled_shape(sps[1], sps[2], 16)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        params_raw = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params_raw)
        common_params_all = params[:, :, :yH, :yW].contiguous()
        _, q_dec, scales_0, means_0 = self.separate_prior(common_params_all, False)
        common_params = self.y_spatial_prior_reduction(common_params_all)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        dtype = means_0.dtype
        device = means_0.device
        B, C, H, W = means_0.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)
        
        y_hat_0 = self.decompress_group_with_mask(scales_0, means_0, mask_0, decoder, cdf, cdf_lengths, offsets)
        y_hat_so_far = y_hat_0

        params_0 = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_1, means_1 = self.y_spatial_prior(self.y_spatial_prior_adaptor_1(params_0)).chunk(2, 1)
        y_hat_1 = self.decompress_group_with_mask(scales_1, means_1, mask_1, decoder, cdf, cdf_lengths, offsets)
        y_hat_so_far = y_hat_so_far + y_hat_1

        params_1 = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_2, means_2 = self.y_spatial_prior(self.y_spatial_prior_adaptor_2(params_1)).chunk(2, 1)
        y_hat_2 = self.decompress_group_with_mask(scales_2, means_2, mask_2, decoder, cdf, cdf_lengths, offsets)
        y_hat_so_far = y_hat_so_far + y_hat_2

        params_2 = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_3, means_3 = self.y_spatial_prior(self.y_spatial_prior_adaptor_3(params_2)).chunk(2, 1)
        y_hat_3 = self.decompress_group_with_mask(scales_3, means_3, mask_3, decoder, cdf, cdf_lengths, offsets)
        y_hat = y_hat_so_far + y_hat_3

        y_hat = y_hat * q_dec
        cudnn.deterministic = False
        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)

        torch.cuda.synchronize()
        end_time = time.process_time()
        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time            
            }
    
