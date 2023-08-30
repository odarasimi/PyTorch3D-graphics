#include <torch/extension.h>

/* Nvidia T4 Tensor Core GPU: https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/

  +---------------+
 /               /|
/               / |
+--------------+  +
|              | /
|              |/
+--------------+

*/

torch::Tensor cuda_trilinear(torch::Tensor vertex_features, torch::Tensor point){
    const int N = vertex_features.size(0), F = vertex_features.size(2)

    torch::Tensor interpolated_feature = torch::zeros({N, F}, vertex_features.options())
    // determine thread size and shared memory usage after going through the nvidia turing arch
}