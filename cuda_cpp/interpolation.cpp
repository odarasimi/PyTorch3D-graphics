#include <torch/extension.h>
#include <ATen/ATen.h>

torch::Tensor trilinear_kernel(
    const torch::Tensor features,
    const torch::Tensor cube_points
)

torch::Tensor trilinear_interpolation(const torch::Tensor features, const torch::Tensor cube_points){
    return trilinear_kernel(features, cube_points)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("trilinear_interpolation", &trilinear_interpolation);
}
