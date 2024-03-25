#include <torch/extension.h>
#include "dcnv4.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

namespace dcnv4 {
    int64_t cuda_version() {
#ifdef WITH_CUDA
        return CUDA_VERSION;
#else
        return -1;
#endif
    }

    TORCH_LIBRARY_FRAGMENT(dcnv4, m) {
        m.def("_cuda_version", &cuda_version);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("_cuda_version", &cuda_version, "get cuda version");
        m.def("dcnv4_forward", &ops::dcnv4_forward, "dcnv4_forward");
        m.def("dcnv4_backward", &ops::detail::_dcnv4_backward, "dcnv4_backward");
        m.def("flash_deform_attn_forward", &ops::flash_deform_attn_forward, "flash_deform_attn_forward");
        m.def("flash_deform_attn_backward", &ops::detail::_flash_deform_attn_backward, "flash_deform_attn_backward");
    }
}
