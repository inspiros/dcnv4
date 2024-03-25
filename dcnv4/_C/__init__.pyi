from typing import Tuple

from torch import Tensor


def _cuda_version() -> int: ...


def dcnv4_forward(value: Tensor, p_offset: Tensor,
                  kernel_h: int, kernel_w: int,
                  stride_h: int, stride_w: int,
                  pad_h: int, pad_w: int,
                  dilation_h: int, dilation_w: int,
                  group: int, group_channels: int,
                  offset_scale: float, im2col_step: int, remove_center: int,
                  d_stride: int, block_thread: int, softmax: bool) -> Tensor: ...


def dcnv4_backward(value: Tensor, p_offset: Tensor,
                   kernel_h: int, kernel_w: int,
                   stride_h: int, stride_w: int,
                   pad_h: int, pad_w: int,
                   dilation_h: int, dilation_w: int,
                   group: int, group_channels: int,
                   offset_scale: float, im2col_step: int, grad_output: Tensor, remove_center: int,
                   d_stride: int, block_thread: int, softmax: bool) -> Tuple[Tensor, Tensor]: ...


def flash_deform_attn_forward(value: Tensor, spatial_shapes: Tensor,
                              level_start_index: Tensor, sampling_loc_attn: Tensor,
                              im2col_step: int, K: int,
                              d_stride: int, block_thread: int) -> Tensor: ...


def flash_deform_attn_backward(value: Tensor, spatial_shapes: Tensor,
                               level_start_index: Tensor, sampling_loc_attn: Tensor,
                               grad_output: Tensor,
                               im2col_step: int, K: int,
                               d_stride: int, block_thread: int) -> Tuple[Tensor, Tensor]: ...
