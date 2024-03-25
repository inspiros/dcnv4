#pragma once

#include <ATen/ATen.h>

namespace dcnv4 {
    namespace ops {
        at::Tensor dcnv4_forward(
                const at::Tensor &value,
                const at::Tensor &p_offset,
                const int kernel_h, const int kernel_w, const int stride_h,
                const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
                const int dilation_w, const int group, const int group_channels,
                const float offset_scale, const int im2col_step, const int remove_center,
                const int d_stride, const int block_thread, const bool softmax);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> _dcnv4_backward(
                    const at::Tensor &value,
                    const at::Tensor &p_offset,
                    const int kernel_h, const int kernel_w, const int stride_h,
                    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
                    const int dilation_w, const int group, const int group_channels,
                    const float offset_scale, const int im2col_step, const at::Tensor &grad_output,
                    const int remove_center, const int d_stride, const int block_thread,
                    const bool softmax);
        }
    }
}
