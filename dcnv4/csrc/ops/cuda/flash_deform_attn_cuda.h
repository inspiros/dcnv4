/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#pragma once

#include <ATen/ATen.h>

namespace dcnv4 {
    namespace ops {
        namespace cuda {
            at::Tensor flash_deform_attn_cuda_forward(
                    const at::Tensor &value, const at::Tensor &spatial_shapes,
                    const at::Tensor &level_start_index, const at::Tensor &sampling_loc_attn,
                    const int im2col_step, const int K, const int d_stride, const int block_thread);

            std::tuple<at::Tensor, at::Tensor> flash_deform_attn_cuda_backward(
                    const at::Tensor &value, const at::Tensor &spatial_shapes,
                    const at::Tensor &level_start_index, const at::Tensor &sampling_loc_attn,
                    const at::Tensor &grad_output, const int im2col_step, const int K,
                    const int d_stride, const int block_thread);
        }
    }
}
