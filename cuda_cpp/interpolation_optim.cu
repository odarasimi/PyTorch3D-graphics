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

//-------------------------------------CALLED AND EXECUTED ON GPU-------------------------------------------------------------------------
/*
__device__ float doubleValue(float x)	
{	
	return x/2;	
}
*/

//-----------------------------------CALLED ON CPU, SPMD EXECUTION ON GPU-----------------------------------------------------------------
template <typename float>
__global__ void trilinear_kernel(
    //difference with Accessor is that a Packed Accessor copies size and stride data inside of its structure instead of pointing to it
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> vertex_features,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> interpolated_feature
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n<feats.size(0) || f<feats.size(2))
        
        const scalar_t u = (points[n][0]+1)/2;
        const scalar_t v = (points[n][1]+1)/2;
        const scalar_t w = (points[n][2]+1)/2;
        
        const scalar_t a = (1-v)*(1-w);
        const scalar_t b = (1-v)*w;
        const scalar_t c = v*(1-w);
        const scalar_t d = 1-a-b-c;
        interpolated_feature[n][f] = (1-u)*(a*vertex_features[n][0][f] +
                                b*vertex_features[n][1][f] +
                                c*vertex_features[n][2][f] +
                                d*vertex_features[n][3][f]) + 
                                u*(a*vertex_features[n][4][f] +
                                b*vertex_features[n][5][f] +
                                c*vertex_features[n][6][f] +
                                d*vertex_features[n][7][f]);
}




//--------------------------------------SERIAL EXECUTION ON CPU-----------------------------------------------------------------
torch::Tensor cuda_trilinear(torch::Tensor vertex_features, torch::Tensor points){

    // const variables N & F to determine the shape of our output dimensions
    const int N = vertex_features.size(0), F = vertex_features.size(2);

    //tensor containing our 2 output dimensions
    torch::Tensor interpolated_feature = torch::zeros({N, F}, vertex_features.options()); 

    // determine thread & block shape 
    const int threads(16, 16);
    //dim3: data structure that encapsulates three unsigned integers: x, y, and z; z defaults to 1
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y) 

    // Kernel launch; AT_DISPATCH... = pytorch macro for dispatching operations based on the data type of the torch tensor 
    // packed accessor = tensor type conversion; 3 and 2 = num of dimensions
    AT_DISPATCH_FLOATING_TYPES(vertex_features.type(), "cuda_trilinear", 
    ([&] {
        trilinear_kernel<<<blocks, threads>>>(
            vertex_features.packed_accessor<float, 3, torch::RestrictPtrTraits>(),
            points.packed_accessor<float, 2, torch::RestrictPtrTraits>(),
            interpolated_feature.packed_accessor<float, 2, torch::RestrictPtrTraits>()
        );
    }));

    return interpolated_feature;
}
