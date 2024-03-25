#pragma once

#include <ATen/ATen.h>

namespace dcnv4 {
    namespace ops {
        at::Tensor flash_deform_attn_forward(
                const at::Tensor &value,
                const at::Tensor &spatial_shapes,
                const at::Tensor &level_start_index,
                const at::Tensor &sampling_loc_attn,
                const int im2col_step, const int K,
                const int d_stride, const int block_thread);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> _flash_deform_attn_backward(
                    const at::Tensor &value,
                    const at::Tensor &spatial_shapes,
                    const at::Tensor &level_start_index,
                    const at::Tensor &sampling_loc_attn,
                    const at::Tensor &grad_output,
                    const int im2col_step,
                    const int K,
                    const int d_stride, const int block_thread);
        }
    }
}
