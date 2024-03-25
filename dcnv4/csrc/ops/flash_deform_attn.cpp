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

#ifdef WITH_CUDA

#include "cuda/flash_deform_attn_cuda.h"

#endif

namespace dcnv4 {
    namespace ops {
        at::Tensor flash_deform_attn_forward(
                const at::Tensor &value,
                const at::Tensor &spatial_shapes,
                const at::Tensor &level_start_index,
                const at::Tensor &sampling_loc_attn,
                const int im2col_step, const int K,
                const int d_stride, const int block_thread) {
            if (value.device().is_cuda()) {
#ifdef WITH_CUDA
                return cuda::flash_deform_attn_cuda_forward(
                        value, spatial_shapes,
                        level_start_index,
                        sampling_loc_attn, im2col_step,
                        K, d_stride, block_thread);
#else
                AT_ERROR("Not compiled with GPU support");
#endif
            }
            AT_ERROR("Not implemented on the CPU");
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> _flash_deform_attn_backward(
                    const at::Tensor &value,
                    const at::Tensor &spatial_shapes,
                    const at::Tensor &level_start_index,
                    const at::Tensor &sampling_loc_attn,
                    const at::Tensor &grad_output,
                    const int im2col_step,
                    const int K,
                    const int d_stride, const int block_thread) {
                if (value.device().is_cuda()) {
#ifdef WITH_CUDA
                    return cuda::flash_deform_attn_cuda_backward(
                            value, spatial_shapes,
                            level_start_index,
                            sampling_loc_attn,
                            grad_output,
                            im2col_step,
                            K, d_stride, block_thread);
#else
                    AT_ERROR("Not compiled with GPU support");
#endif
                }
                AT_ERROR("Not implemented on the CPU");
            }
        }
    }
}
