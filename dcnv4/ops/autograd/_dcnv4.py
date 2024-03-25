from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from dcnv4._C import dcnv4_forward, dcnv4_backward
from .table import TABLE, BWDTABLE

__all__ = [
    'DCNv4Function',
    'dcnv4',
]


def factors(N):
    res = []
    for i in range(1, N + 1):
        if N % i == 0:
            res.append(i)
    return res


def findspec(B, H, W, G, C):
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in TABLE:
        return TABLE[key][0], TABLE[key][1]

    d_stride = 8
    ms = factors(B * H * W)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    key = f"{B}x{H}x{W}x{G}x{C}"
    TABLE[key] = (d_stride, n_thread)
    return d_stride, n_thread


def find_spec_bwd(B, H, W, G, C):
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in BWDTABLE:
        return BWDTABLE[key][0], BWDTABLE[key][1]

    if C >= 64:
        d_stride = 2
    else:
        d_stride = 1

    ms = factors(B * H * W)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 256:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


class DCNv4Function(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, offset_mask,
                kernel_h, kernel_w, stride_h, stride_w,
                pad_h, pad_w, dilation_h, dilation_w,
                group, group_channels, offset_scale,
                im2col_step, remove_center):
        forward_d_stride, forward_block_thread = findspec(
            input.shape[0], input.shape[1], input.shape[2], group, group_channels)
        backward_d_stride, backward_block_thread = find_spec_bwd(
            input.shape[0], input.shape[1], input.shape[2], group, group_channels)

        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center
        ctx.backward_d_stride = backward_d_stride
        ctx.backward_block_thread = backward_block_thread

        output = dcnv4_forward(
            input, offset_mask, kernel_h,
            kernel_w, stride_h, stride_w, pad_h,
            pad_w, dilation_h, dilation_w, group,
            group_channels, offset_scale,
            ctx.im2col_step,
            remove_center,
            forward_d_stride,
            forward_block_thread,
            False)
        ctx.save_for_backward(input, offset_mask)

        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset_mask = ctx.saved_tensors

        grad_input, grad_offset_mask = dcnv4_backward(
            input, offset_mask, ctx.kernel_h,
            ctx.kernel_w, ctx.stride_h, ctx.stride_w, ctx.pad_h,
            ctx.pad_w, ctx.dilation_h, ctx.dilation_w, ctx.group,
            ctx.group_channels, ctx.offset_scale, ctx.im2col_step,
            grad_output.contiguous(), ctx.remove_center,
            ctx.backward_d_stride, ctx.backward_block_thread,
            False)

        return grad_input, grad_offset_mask, \
            None, None, None, None, None, None, None, \
            None, None, None, None, None, None


def dcnv4(input: Tensor, offset_mask: Tensor,
          kernel_h: int, kernel_w: int,
          stride_h: int, stride_w: int,
          pad_h: int, pad_w: int,
          dilation_h: int, dilation_w: int,
          group: int, group_channels: int,
          offset_scale: float,
          im2col_step: int, remove_center: int) -> Tensor:
    return DCNv4Function.apply(input, offset_mask,
                               kernel_h, kernel_w, stride_h, stride_w,
                               pad_h, pad_w, dilation_h, dilation_w,
                               group, group_channels, offset_scale,
                               im2col_step, remove_center)
