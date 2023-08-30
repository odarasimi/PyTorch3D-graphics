#include <torch/extension.h>
#include <ATen/ATen.h>

torch::Tensor trilinear_interpolation(torch::Tensor features, torch::Tensor points){
    CHECK_INPUT(features);
    CHECK_INPUT(points);

    return cuda_trilinear(features, points)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("trilinear_interpolation", &trilinear_interpolation);
}