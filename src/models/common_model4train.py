# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import math
import torch.nn.functional as F
from .entropy_models4train import BitEstimator, GaussianEncoder, EntropyCoder

def restore_y_4x(y, means, mask):
    return (torch.cat((y, y, y, y), dim=1) + means) * mask

class CompressionModel(nn.Module):
    def __init__(self, z_channel):
        super().__init__()

        self.z_channel = z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(64, z_channel)
        self.gaussian_encoder = GaussianEncoder()

        self.masks = {}
        self.cuda_streams = {}

    def get_cuda_stream(self, device, idx=0, priority=0):
        key = f"{device}_{priority}_{idx}"
        if key not in self.cuda_streams:
            self.cuda_streams[key] = torch.cuda.Stream(device, priority=priority)
        return self.cuda_streams[key]

    @staticmethod
    def get_qp_num():
        return 64

    @staticmethod
    def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom

    @staticmethod
    def get_downsampled_shape(height, width, p):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return int(new_h / p + 0.5), int(new_w / p + 0.5)

    def update(self, force_zero_thres=None):
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(self.entropy_coder, force_zero_thres=force_zero_thres)
        self.bit_estimator_z.update(self.entropy_coder)

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.entropy_coder.set_use_two_entropy_coders(use_two_entropy_coders)

    def replicate_pad(self, x, pad_b, pad_r):
        if pad_b == 0 and pad_r == 0:
            return x
        else:
            return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")

    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_r, padding_b = self.get_padding_size(H, W, 4)
        y_pad = self.replicate_pad(y, padding_b, padding_r)
        return y_pad

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
    
    def ste_round(self, x):
        return (torch.round(x) - x).detach() + x

    def process_with_mask(self, y, scales, means, mask, write):
        return self.gaussian_encoder.process_with_mask(y, scales, means, mask, write)

    @staticmethod
    def probs_to_bits(probs):
        factor = -1.0 / math.log(2.0)
        bits = torch.log(torch.clamp(probs, 1e-9, 1.0)) * factor
        # bits = torch.clamp(bits, 0, None) # 截断负值的bits应该是必需的
        return bits
    
    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        # probs = probs.to(torch.float32)
        return CompressionModel.probs_to_bits(probs)
    
    def get_z_bits(self, z, bit_estimator, index):
        probs = bit_estimator.get_cdf(z + 0.5, index) - bit_estimator.get_cdf(z - 0.5, index)
        # probs = probs.to(torch.float32)
        return CompressionModel.probs_to_bits(probs)

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


    @staticmethod
    def single_part_for_writing_4x(x):
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)


    def compress_prior_4x(self, y, common_params, y_spatial_prior_reduction,
                          y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                          y_spatial_prior_adaptor_3, y_spatial_prior, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params, False)
        common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        y = y * q_enc

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0, write)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1, write)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2, write)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3, write)

        y_res = (y_res_0 + y_res_1) + (y_res_2 +y_res_3)
        y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)
        y_hat = (y_hat_so_far + y_hat_3) * q_dec

        if write:
            y_q_w_0 = self.single_part_for_writing_4x(y_q_0)
            y_q_w_1 = self.single_part_for_writing_4x(y_q_1)
            y_q_w_2 = self.single_part_for_writing_4x(y_q_2)
            y_q_w_3 = self.single_part_for_writing_4x(y_q_3)
            s_w_0 = self.single_part_for_writing_4x(s_hat_0)
            s_w_1 = self.single_part_for_writing_4x(s_hat_1)
            s_w_2 = self.single_part_for_writing_4x(s_hat_2)
            s_w_3 = self.single_part_for_writing_4x(s_hat_3)
            return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat
        else:
            return y_res, y_q, y_hat, scales_hat

    def decompress_prior_4x(self, common_params, y_spatial_prior_reduction,
                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                            y_spatial_prior_adaptor_3, y_spatial_prior):
        _, quant_step, scales, means = self.separate_prior(common_params, False)
        common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        scales_r = self.single_part_for_writing_4x(scales * mask_0)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_0)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_1)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_2)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_2)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_3)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_3)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * quant_step

        return y_hat
