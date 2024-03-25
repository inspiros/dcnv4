import math
import torch

import dcnv4


def test_dcnv4(dtype=torch.float16,
               device='cuda'):
    torch.manual_seed(3)
    device = torch.device(device)

    batch_sz, group, group_channels = 1, 4, 8
    in_h, in_w = 16, 16
    kernel_h, kernel_w = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation_h, dilation_w = 1, 1
    offset_scale = 2.0
    im2col_step = 128
    remove_center = False
    p = kernel_h * kernel_w - remove_center

    out_h = (in_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (in_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    input = torch.randn(batch_sz, in_h, in_w, group * group_channels,
                        dtype=dtype, device=device, requires_grad=True)
    offset = torch.randn(batch_sz, out_h, out_w, group * p * 2,
                         dtype=dtype, device=device, requires_grad=True)
    offset.data.mul_(10)
    mask = torch.rand(batch_sz, out_h, out_w, group, p,
                      dtype=dtype, device=device, requires_grad=True)
    mask.data.add_(1e-5)
    offset_mask = torch.cat(
        [offset.unflatten(-1, (group, p * 2)), mask], dim=-1).flatten(-2)

    optim = torch.optim.Optimizer([input, offset, mask], {})

    def pad(om):
        padded_zero = int(math.ceil(om.shape[3]/8)*8) - om.shape[3]
        padded = om.new_zeros(om.shape[0], om.shape[1], om.shape[2], padded_zero)
        return torch.cat([om, padded], dim=-1)

    c_output = dcnv4.ops.dcnv4(input, pad(offset_mask),
                               kernel_h, kernel_w,
                               stride_h, stride_w,
                               kernel_h // 2, kernel_w // 2,
                               dilation_h, dilation_w,
                               group, group_channels, offset_scale,
                               im2col_step, remove_center)
    c_output.sum().backward()
    optim.zero_grad()
    print(c_output)


def test_flash_deform_attn(dtype=torch.float16,
                           device='cuda'):
    torch.manual_seed(3)
    device = torch.device(device)

    batch_sz, m, d = 1, 4, 8
    lq, l, p = 100 * 152, 4, 8
    im2col_step = 128

    shapes = torch.tensor([[100, 152], [ 50,  76], [ 25,  38], [ 13,  19]], dtype=torch.long, device=device)
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    s = shapes.prod(1).sum().item()

    value = torch.randn(batch_sz, s, m, d,
                        dtype=dtype, device=device, requires_grad=True)
    value.data.mul_(0.01)
    sampling_locations = torch.rand(batch_sz, lq, m, l, p, 2,
                                    dtype=dtype, device=device)
    attention_weights = torch.rand(batch_sz, lq, m, l, p,
                                   dtype=dtype, device=device)
    attention_weights = torch.nn.functional.softmax(
        attention_weights.flatten(-2, -1), dim=-1).unflatten(-1, (l, p))
    attention_weights.requires_grad_(True)
    sampling_loc_attn = torch.cat([sampling_locations.reshape(batch_sz, lq, m, l*p*2),
                                   attention_weights.view(batch_sz, lq, m, l*p)], dim=-1)

    optim = torch.optim.Optimizer([value, attention_weights], {})

    c_output = dcnv4.ops.flash_deform_attn(value,
                                           shapes,
                                           level_start_index,
                                           sampling_loc_attn,
                                           im2col_step,
                                           p)
    c_output.sum().backward()
    optim.zero_grad()
    print(c_output)


if __name__ == '__main__':
    test_dcnv4()
    test_flash_deform_attn()
