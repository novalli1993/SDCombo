/*!
**************************************************************************************************
* InternImage
* Copyright (c) 2022 OpenGVLab
* Licensed under The MIT License [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "cuda/dcnv3_im2col_cuda.cuh"
#include <vector>

#include <ATen/ATen.h>
// [SDCombo-patch] 原为：
//   #include <ATen/cuda/CUDAContext.h>  -> 链式拉入 cusparse.h（conda 工具链无此头）
//   #include <torch/torch.h>             -> 在 nvcc 下与 libcu++ 的 std 命名空间冲突
// 本文件只需要 getCurrentCUDAStream()，由 c10/cuda/CUDAStream.h 直接提供
// （注意不能用 ATen/cuda/CUDAContextLight.h，它同样会 include cusparse.h）。
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

at::Tensor dcnv3_cuda_forward(const at::Tensor &input, const at::Tensor &offset,
                              const at::Tensor &mask, const int kernel_h,
                              const int kernel_w, const int stride_h,
                              const int stride_w, const int pad_h,
                              const int pad_w, const int dilation_h,
                              const int dilation_w, const int group,
                              const int group_channels,
                              const float offset_scale, const int im2col_step) {
    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(offset.is_contiguous(), "offset tensor has to be contiguous");
    AT_ASSERTM(mask.is_contiguous(), "mask tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");

    const int batch = input.size(0);
    const int height_in = input.size(1);
    const int width_in = input.size(2);
    const int channels = input.size(3);
    const int height_out =
        (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
        1;
    const int width_out =
        (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
        1;
    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0,
               "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);
    AT_ASSERTM(
        channels == (group * group_channels),
        "Input channels and group times group channels wont match: (%d vs %d).",
        channels, group * group_channels);

    auto output =
        at::zeros({batch, height_out, width_out, group * group_channels},
                  input.options());

    const int batch_n = im2col_step_;
    auto output_n = output.view({batch / batch_n, batch_n, height_out,
                                 width_out, group * group_channels});
    auto per_input_size = height_in * width_in * group * group_channels;
    auto per_offset_size =
        height_out * width_out * group * kernel_h * kernel_w * 2;
    auto per_mask_size = height_out * width_out * group * kernel_h * kernel_w;
    for (int n = 0; n < batch / im2col_step_; ++n) {
        auto columns = output_n.select(0, n);
        // AT_DISPATCH_FLOATING_TYPES(
        // [SDCombo-patch] input.type() 返回 at::DeprecatedTypeProperties，
        // PyTorch 2.x 的 AT_DISPATCH_* 需要 c10::ScalarType，故改用 scalar_type()。
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
                dcnv3_im2col_cuda(
                    at::cuda::getCurrentCUDAStream(),
                    input.data<scalar_t>() + n * im2col_step_ * per_input_size,
                    offset.data<scalar_t>() +
                        n * im2col_step_ * per_offset_size,
                    mask.data<scalar_t>() + n * im2col_step_ * per_mask_size,
                    columns.data<scalar_t>(), kernel_h, kernel_w, stride_h,
                    stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                    group_channels, batch_n, height_in, width_in, height_out,
                    width_out, offset_scale);
            }));
    }

    return output;
}

std::vector<at::Tensor>
dcnv3_cuda_backward(const at::Tensor &input, const at::Tensor &offset,
                    const at::Tensor &mask, const int kernel_h,
                    const int kernel_w, const int stride_h, const int stride_w,
                    const int pad_h, const int pad_w, const int dilation_h,
                    const int dilation_w, const int group,
                    const int group_channels, const float offset_scale,
                    const at::Tensor &grad_output, const int im2col_step) {

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(offset.is_contiguous(), "offset tensor has to be contiguous");
    AT_ASSERTM(mask.is_contiguous(), "mask tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(),
               "grad_output tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");
    AT_ASSERTM(grad_output.type().is_cuda(),
               "grad_output must be a CUDA tensor");

    const int batch = input.size(0);
    const int height_in = input.size(1);
    const int width_in = input.size(2);
    const int channels = input.size(3);
    const int height_out =
        (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
        1;
    const int width_out =
        (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
        1;
    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0,
               "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);
    AT_ASSERTM(
        channels == (group * group_channels),
        "Input channels and group times group channels wont match: (%d vs %d).",
        channels, group * group_channels);

    auto dtype = input.dtype();
    if (dtype == at::kHalf) {
        dtype = at::kFloat;
    }

    auto grad_input = at::zeros_like(input, dtype);
    auto grad_offset = at::zeros_like(offset, dtype);
    auto grad_mask = at::zeros_like(mask, dtype);

    const int batch_n = im2col_step_;
    auto per_input_size = height_in * width_in * group * group_channels;
    auto per_offset_size =
        height_out * width_out * group * kernel_h * kernel_w * 2;
    auto per_mask_size = height_out * width_out * group * kernel_h * kernel_w;
    auto grad_output_n =
        grad_output.view({batch / im2col_step_, batch_n, height_out * width_out,
                          group, group_channels});

    for (int n = 0; n < batch / im2col_step_; ++n) {
        auto grad_output_g = grad_output_n.select(0, n);
        // AT_DISPATCH_FLOATING_TYPES(
        // [SDCombo-patch] 同上：input.type() -> input.scalar_type()
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
                dcnv3_col2im_cuda(
                    at::cuda::getCurrentCUDAStream(),
                    grad_output_g.data<scalar_t>(),
                    input.data<scalar_t>() + n * im2col_step_ * per_input_size,
                    offset.data<scalar_t>() +
                        n * im2col_step_ * per_offset_size,
                    mask.data<scalar_t>() + n * im2col_step_ * per_mask_size,
                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, group, group_channels, batch_n,
                    height_in, width_in, height_out, width_out, offset_scale,
                    grad_input.data<opmath_t>() +
                        n * im2col_step_ * per_input_size,
                    grad_offset.data<opmath_t>() +
                        n * im2col_step_ * per_offset_size,
                    grad_mask.data<opmath_t>() +
                        n * im2col_step_ * per_mask_size);
            }));
    }

    // [SDCombo-patch] 原为 torch::kHalf（依赖 <torch/torch.h>，在 nvcc 下会与
    // libcu++ 的 std 命名空间冲突），改用 ATen 的 at::kHalf。
    if (input.dtype() == at::kHalf) {
        return {grad_input.to(at::kHalf), grad_offset.to(at::kHalf),
                grad_mask.to(at::kHalf)};
    } else {
        return {grad_input, grad_offset, grad_mask};
    }
}