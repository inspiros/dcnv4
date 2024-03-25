import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from dcnv4._C import flash_deform_attn_forward, flash_deform_attn_backward

__all__ = [
    'FlashDeformAttnFunction',
    'flash_deform_attn',
]


def _compute_shm_size_cap():
    shm_size_dict = {
        "8.0": 163000,
        "8.6": 99000,
        "8.7": 163000,
        "8.9": 99000,
        "9.0": 227000,
        "7.5": 64000,
        "7.0": 96000,
    }
    cuda_capability = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
    return shm_size_dict.get(cuda_capability, None)


def factors(N):
    res = []
    for i in range(1, N + 1):
        if N % i == 0:
            res.append(i)
    return res


def findspec(B, Q, G, C):
    d_stride = 8
    ms = factors(B * Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


def findspec_bwd(B, Q, G, C):
    if C >= 64:
        d_stride = 2
    else:
        d_stride = 1

    ms = factors(B * Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 256:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


class FlashDeformAttnFunction(Function):
    @staticmethod
    @custom_fwd
    @torch.autocast("cuda", enabled=True, dtype=torch.float16)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_loc_attn, im2col_step, K=8):
        ctx.im2col_step = im2col_step
        ctx.K = K
        d_stride, blockthread = findspec(
            value.shape[0], sampling_loc_attn.shape[1], value.shape[2], value.shape[3])
        d_stride_backward, blockthread_backward = findspec_bwd(
            value.shape[0], sampling_loc_attn.shape[1], value.shape[2], value.shape[3])

        ctx.d_stride_backward = d_stride_backward
        ctx.blockthread_backward = blockthread_backward

        output = flash_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
            ctx.im2col_step,
            K,
            d_stride,
            blockthread)

        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_loc_attn)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_loc_attn = ctx.saved_tensors

        grad_value, grad_sampling_loc_attn = flash_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
            grad_output.contiguous(),
            ctx.im2col_step,
            ctx.K,
            ctx.d_stride_backward,
            ctx.blockthread_backward)

        return grad_value, None, None, grad_sampling_loc_attn, None, None


def flash_deform_attn(value: Tensor,
                      value_spatial_shapes: Tensor,
                      value_level_start_index: Tensor,
                      sampling_loc_attn: Tensor,
                      im2col_step: int, K: int = 8):
    return FlashDeformAttnFunction.apply(value, value_spatial_shapes, value_level_start_index,
                                         sampling_loc_attn, im2col_step, K)
