#include "network.hpp"
#include "nnet.hpp"

namespace AlphaGomoku {
namespace White {
QuantizedNetwork::QuantizedNetwork()
    :     functional_1_1_conv2d_9_1_BiasAdd_quant(functional_1_1_conv2d_9_1_convolution_ReadVariableOp_0_quantized, functional_1_1_conv2d_9_1_Squeeze_0_quantized),
    functional_1_1_batch_normalization_9_1_batchnorm_mul_1_quant(functional_1_1_batch_normalization_9_1_batchnorm_mul_0_quantized),
    functional_1_1_batch_normalization_9_1_batchnorm_add_1_quant(functional_1_1_batch_normalization_9_1_batchnorm_sub_0_quantized),
    functional_1_1_conv2d_10_1_BiasAdd_quant(functional_1_1_conv2d_10_1_convolution_ReadVariableOp_0_quantized, functional_1_1_conv2d_10_1_Squeeze_0_quantized),
    functional_1_1_batch_normalization_10_1_batchnorm_mul_1_quant(functional_1_1_batch_normalization_10_1_batchnorm_mul_0_quantized),
    functional_1_1_batch_normalization_10_1_batchnorm_add_1_quant(functional_1_1_batch_normalization_10_1_batchnorm_sub_0_quantized),
    functional_1_1_conv2d_11_1_BiasAdd_quant(functional_1_1_conv2d_11_1_convolution_ReadVariableOp_0_quantized, functional_1_1_conv2d_11_1_Squeeze_0_quantized),
    functional_1_1_batch_normalization_11_1_batchnorm_mul_1_quant(functional_1_1_batch_normalization_11_1_batchnorm_mul_0_quantized),
    functional_1_1_batch_normalization_11_1_batchnorm_add_1_quant(functional_1_1_batch_normalization_11_1_batchnorm_sub_0_quantized),
    functional_1_1_conv2d_12_1_BiasAdd_quant(functional_1_1_conv2d_12_1_convolution_ReadVariableOp_0_quantized, functional_1_1_conv2d_12_1_Squeeze_0_quantized),
    functional_1_1_batch_normalization_12_1_batchnorm_mul_1_quant(functional_1_1_batch_normalization_12_1_batchnorm_mul_0_quantized),
    functional_1_1_batch_normalization_12_1_batchnorm_add_1_quant(functional_1_1_batch_normalization_12_1_batchnorm_sub_0_quantized),
    functional_1_1_conv2d_13_1_BiasAdd_quant(functional_1_1_conv2d_13_1_convolution_ReadVariableOp_0_quantized, functional_1_1_conv2d_13_1_Squeeze_0_quantized),
    functional_1_1_batch_normalization_13_1_batchnorm_mul_1_quant(functional_1_1_batch_normalization_13_1_batchnorm_mul_0_quantized),
    functional_1_1_batch_normalization_13_1_batchnorm_add_1_quant(functional_1_1_batch_normalization_13_1_batchnorm_sub_0_quantized),
    functional_1_1_conv2d_14_1_BiasAdd_quant(functional_1_1_conv2d_14_1_convolution_ReadVariableOp_0_quantized, functional_1_1_conv2d_14_1_Squeeze_0_quantized),
    functional_1_1_batch_normalization_14_1_batchnorm_mul_1_quant(functional_1_1_batch_normalization_14_1_batchnorm_mul_0_quantized),
    functional_1_1_batch_normalization_14_1_batchnorm_add_1_quant(functional_1_1_batch_normalization_14_1_batchnorm_sub_0_quantized),
    functional_1_1_conv2d_15_1_BiasAdd_quant(functional_1_1_conv2d_15_1_convolution_ReadVariableOp_0_quantized, functional_1_1_conv2d_15_1_Squeeze_0_quantized),
    functional_1_1_batch_normalization_15_1_batchnorm_mul_1_quant(functional_1_1_batch_normalization_15_1_batchnorm_mul_0_quantized),
    functional_1_1_batch_normalization_15_1_batchnorm_add_1_quant(functional_1_1_batch_normalization_15_1_batchnorm_sub_0_quantized),
    functional_1_1_conv2d_17_1_convolution_quant(functional_1_1_conv2d_17_1_convolution_ReadVariableOp_0_quantized, ConvAddFusion_Add_B_functional_1_1_conv2d_17_1_Reshape_0_quantized),
    functional_1_1_conv2d_16_1_BiasAdd_quant(functional_1_1_conv2d_16_1_convolution_ReadVariableOp_0_quantized, functional_1_1_conv2d_16_1_Squeeze_0_quantized),
    functional_1_1_batch_normalization_17_1_batchnorm_mul_1_quant(functional_1_1_batch_normalization_17_1_batchnorm_mul_0_quantized),
    functional_1_1_batch_normalization_16_1_batchnorm_mul_1_quant(functional_1_1_batch_normalization_16_1_batchnorm_mul_0_quantized),
    functional_1_1_batch_normalization_17_1_batchnorm_add_1_quant(functional_1_1_batch_normalization_17_1_batchnorm_sub_0_quantized),
    functional_1_1_batch_normalization_16_1_batchnorm_add_1_quant(functional_1_1_batch_normalization_16_1_batchnorm_sub_0_quantized),
    gemm_quant(functional_1_1_dense_4_1_Cast_ReadVariableOp_0_quantized, functional_1_1_dense_4_1_BiasAdd_ReadVariableOp_0_quantized),
    gemm_token_1_quant(functional_1_1_dense_3_1_Cast_ReadVariableOp_0_quantized, functional_1_1_dense_3_1_BiasAdd_ReadVariableOp_0_quantized),
    gemm_token_0_quant(functional_1_1_dense_5_1_Cast_ReadVariableOp_0_quantized, functional_1_1_dense_5_1_Add_ReadVariableOp_0_quantized)
{}

AlphaGomoku::GodNet::RetType QuantizedNetwork::feed(const Tensor<uint8_t, 3, 15, 15>& input){
    // Direct chaining approach - no intermediate tensor tracking
    
    // Unknown operation: QuantizeLinear
    functional_1_1_conv2d_9_1_BiasAdd_quant.feed(input);

    functional_1_1_batch_normalization_9_1_batchnorm_mul_1_quant.mul(functional_1_1_conv2d_9_1_BiasAdd_quant.output());
    functional_1_1_batch_normalization_9_1_batchnorm_add_1_quant.add(functional_1_1_conv2d_9_1_BiasAdd_quant.output());
    functional_1_1_conv2d_10_1_BiasAdd_quant.feed(functional_1_1_conv2d_9_1_BiasAdd_quant.output());

    functional_1_1_batch_normalization_10_1_batchnorm_mul_1_quant.mul(functional_1_1_conv2d_10_1_BiasAdd_quant.output());
    functional_1_1_batch_normalization_10_1_batchnorm_add_1_quant.add(functional_1_1_conv2d_10_1_BiasAdd_quant.output());
    functional_1_1_conv2d_11_1_BiasAdd_quant.feed(functional_1_1_conv2d_10_1_BiasAdd_quant.output());

    functional_1_1_batch_normalization_11_1_batchnorm_mul_1_quant.mul(functional_1_1_conv2d_11_1_BiasAdd_quant.output());
    functional_1_1_batch_normalization_11_1_batchnorm_add_1_quant.add(functional_1_1_conv2d_11_1_BiasAdd_quant.output());
    QElemWise</* a_scale */ functional_1_1_batch_normalization_11_1_batchnorm_add_1_0_scale, /* a_zp */ functional_1_1_batch_normalization_11_1_batchnorm_add_1_0_zero_point, /* b_scale */ functional_1_1_batch_normalization_9_1_batchnorm_add_1_0_scale, /* b_zp */ functional_1_1_batch_normalization_9_1_batchnorm_add_1_0_zero_point, /* c_scale */ functional_1_1_add_3_1_Add_0_scale, /* c_zp */ functional_1_1_add_3_1_Add_0_zero_point, /* C */ 32, /* H */ 15, /* W */ 15>::add(functional_1_1_conv2d_11_1_BiasAdd_quant.output(), functional_1_1_conv2d_9_1_BiasAdd_quant.output());
    functional_1_1_conv2d_12_1_BiasAdd_quant.feed(functional_1_1_conv2d_11_1_BiasAdd_quant.output());

    functional_1_1_batch_normalization_12_1_batchnorm_mul_1_quant.mul(functional_1_1_conv2d_12_1_BiasAdd_quant.output());
    functional_1_1_batch_normalization_12_1_batchnorm_add_1_quant.add(functional_1_1_conv2d_12_1_BiasAdd_quant.output());
    functional_1_1_conv2d_13_1_BiasAdd_quant.feed(functional_1_1_conv2d_12_1_BiasAdd_quant.output());

    functional_1_1_batch_normalization_13_1_batchnorm_mul_1_quant.mul(functional_1_1_conv2d_13_1_BiasAdd_quant.output());
    functional_1_1_batch_normalization_13_1_batchnorm_add_1_quant.add(functional_1_1_conv2d_13_1_BiasAdd_quant.output());
    QElemWise</* a_scale */ functional_1_1_batch_normalization_13_1_batchnorm_add_1_0_scale, /* a_zp */ functional_1_1_batch_normalization_13_1_batchnorm_add_1_0_zero_point, /* b_scale */ functional_1_1_add_3_1_Add_0_scale, /* b_zp */ functional_1_1_add_3_1_Add_0_zero_point, /* c_scale */ functional_1_1_add_4_1_Add_0_scale, /* c_zp */ functional_1_1_add_4_1_Add_0_zero_point, /* C */ 32, /* H */ 15, /* W */ 15>::add(functional_1_1_conv2d_13_1_BiasAdd_quant.output(), functional_1_1_conv2d_11_1_BiasAdd_quant.output());
    functional_1_1_conv2d_14_1_BiasAdd_quant.feed(functional_1_1_conv2d_13_1_BiasAdd_quant.output());

    functional_1_1_batch_normalization_14_1_batchnorm_mul_1_quant.mul(functional_1_1_conv2d_14_1_BiasAdd_quant.output());
    functional_1_1_batch_normalization_14_1_batchnorm_add_1_quant.add(functional_1_1_conv2d_14_1_BiasAdd_quant.output());
    functional_1_1_conv2d_15_1_BiasAdd_quant.feed(functional_1_1_conv2d_14_1_BiasAdd_quant.output());

    functional_1_1_batch_normalization_15_1_batchnorm_mul_1_quant.mul(functional_1_1_conv2d_15_1_BiasAdd_quant.output());
    functional_1_1_batch_normalization_15_1_batchnorm_add_1_quant.add(functional_1_1_conv2d_15_1_BiasAdd_quant.output());
    QElemWise</* a_scale */ functional_1_1_batch_normalization_15_1_batchnorm_add_1_0_scale, /* a_zp */ functional_1_1_batch_normalization_15_1_batchnorm_add_1_0_zero_point, /* b_scale */ functional_1_1_add_4_1_Add_0_scale, /* b_zp */ functional_1_1_add_4_1_Add_0_zero_point, /* c_scale */ functional_1_1_add_5_1_Add_0_scale, /* c_zp */ functional_1_1_add_5_1_Add_0_zero_point, /* C */ 32, /* H */ 15, /* W */ 15>::add(functional_1_1_conv2d_15_1_BiasAdd_quant.output(), functional_1_1_conv2d_13_1_BiasAdd_quant.output());
    functional_1_1_conv2d_17_1_convolution_quant.feed(functional_1_1_conv2d_15_1_BiasAdd_quant.output());

    functional_1_1_conv2d_16_1_BiasAdd_quant.feed(functional_1_1_conv2d_15_1_BiasAdd_quant.output());

    functional_1_1_batch_normalization_17_1_batchnorm_mul_1_quant.mul(functional_1_1_conv2d_17_1_convolution_quant.output());
    functional_1_1_batch_normalization_16_1_batchnorm_mul_1_quant.mul(functional_1_1_conv2d_16_1_BiasAdd_quant.output());
    functional_1_1_batch_normalization_17_1_batchnorm_add_1_quant.add(functional_1_1_conv2d_17_1_convolution_quant.output());
    functional_1_1_batch_normalization_16_1_batchnorm_add_1_quant.add(functional_1_1_conv2d_16_1_BiasAdd_quant.output());
    functional_1_1_conv2d_17_1_convolution_quant.output().flatten(); // Reshape: functional_1_1/flatten_3_1/Reshape
    functional_1_1_conv2d_16_1_BiasAdd_quant.output().flatten(); // Reshape: functional_1_1/flatten_2_1/Reshape
    gemm_quant.feed(functional_1_1_conv2d_17_1_convolution_quant.output().flatten());

    gemm_token_1_quant.feed(functional_1_1_conv2d_16_1_BiasAdd_quant.output().flatten());

    gemm_token_0_quant.feed(gemm_quant.output());

    // Unknown operation: DequantizeLinear
    // Unknown operation: DequantizeLinear
    pi = softmax(gemm_token_1_quant.output()); // Softmax activation
    v = Tanh::call(gemm_token_0_quant.output()[0]); // Tanh activation

    return output();
}

} // namespace White


namespace Black {

QuantizedNetwork::QuantizedNetwork()
    :     functional_1_conv2d_1_BiasAdd_quant(functional_1_conv2d_1_convolution_ReadVariableOp_0_quantized, functional_1_conv2d_1_Squeeze_0_quantized),
    functional_1_batch_normalization_1_batchnorm_mul_1_quant(functional_1_batch_normalization_1_batchnorm_mul_0_quantized),
    functional_1_batch_normalization_1_batchnorm_add_1_quant(functional_1_batch_normalization_1_batchnorm_sub_0_quantized),
    functional_1_conv2d_1_2_BiasAdd_quant(functional_1_conv2d_1_2_convolution_ReadVariableOp_0_quantized, functional_1_conv2d_1_2_Squeeze_0_quantized),
    functional_1_batch_normalization_1_2_batchnorm_mul_1_quant(functional_1_batch_normalization_1_2_batchnorm_mul_0_quantized),
    functional_1_batch_normalization_1_2_batchnorm_add_1_quant(functional_1_batch_normalization_1_2_batchnorm_sub_0_quantized),
    functional_1_conv2d_2_1_BiasAdd_quant(functional_1_conv2d_2_1_convolution_ReadVariableOp_0_quantized, functional_1_conv2d_2_1_Squeeze_0_quantized),
    functional_1_batch_normalization_2_1_batchnorm_mul_1_quant(functional_1_batch_normalization_2_1_batchnorm_mul_0_quantized),
    functional_1_batch_normalization_2_1_batchnorm_add_1_quant(functional_1_batch_normalization_2_1_batchnorm_sub_0_quantized),
    functional_1_conv2d_3_1_BiasAdd_quant(functional_1_conv2d_3_1_convolution_ReadVariableOp_0_quantized, functional_1_conv2d_3_1_Squeeze_0_quantized),
    functional_1_batch_normalization_3_1_batchnorm_mul_1_quant(functional_1_batch_normalization_3_1_batchnorm_mul_0_quantized),
    functional_1_batch_normalization_3_1_batchnorm_add_1_quant(functional_1_batch_normalization_3_1_batchnorm_sub_0_quantized),
    functional_1_conv2d_4_1_BiasAdd_quant(functional_1_conv2d_4_1_convolution_ReadVariableOp_0_quantized, functional_1_conv2d_4_1_Squeeze_0_quantized),
    functional_1_batch_normalization_4_1_batchnorm_mul_1_quant(functional_1_batch_normalization_4_1_batchnorm_mul_0_quantized),
    functional_1_batch_normalization_4_1_batchnorm_add_1_quant(functional_1_batch_normalization_4_1_batchnorm_sub_0_quantized),
    functional_1_conv2d_5_1_BiasAdd_quant(functional_1_conv2d_5_1_convolution_ReadVariableOp_0_quantized, functional_1_conv2d_5_1_Squeeze_0_quantized),
    functional_1_batch_normalization_5_1_batchnorm_mul_1_quant(functional_1_batch_normalization_5_1_batchnorm_mul_0_quantized),
    functional_1_batch_normalization_5_1_batchnorm_add_1_quant(functional_1_batch_normalization_5_1_batchnorm_sub_0_quantized),
    functional_1_conv2d_6_1_BiasAdd_quant(functional_1_conv2d_6_1_convolution_ReadVariableOp_0_quantized, functional_1_conv2d_6_1_Squeeze_0_quantized),
    functional_1_batch_normalization_6_1_batchnorm_mul_1_quant(functional_1_batch_normalization_6_1_batchnorm_mul_0_quantized),
    functional_1_batch_normalization_6_1_batchnorm_add_1_quant(functional_1_batch_normalization_6_1_batchnorm_sub_0_quantized),
    functional_1_conv2d_7_1_BiasAdd_quant(functional_1_conv2d_7_1_convolution_ReadVariableOp_0_quantized, functional_1_conv2d_7_1_Squeeze_0_quantized),
    functional_1_conv2d_8_1_convolution_quant(functional_1_conv2d_8_1_convolution_ReadVariableOp_0_quantized, ConvAddFusion_Add_B_functional_1_conv2d_8_1_Reshape_0_quantized),
    functional_1_batch_normalization_7_1_batchnorm_mul_1_quant(functional_1_batch_normalization_7_1_batchnorm_mul_0_quantized),
    functional_1_batch_normalization_8_1_batchnorm_mul_1_quant(functional_1_batch_normalization_8_1_batchnorm_mul_0_quantized),
    functional_1_batch_normalization_7_1_batchnorm_add_1_quant(functional_1_batch_normalization_7_1_batchnorm_sub_0_quantized),
    functional_1_batch_normalization_8_1_batchnorm_add_1_quant(functional_1_batch_normalization_8_1_batchnorm_sub_0_quantized),
    gemm_quant(functional_1_dense_3_Cast_ReadVariableOp_0_quantized, functional_1_dense_3_BiasAdd_ReadVariableOp_0_quantized),
    gemm_token_0_quant(functional_1_dense_1_1_Cast_ReadVariableOp_0_quantized, functional_1_dense_1_1_BiasAdd_ReadVariableOp_0_quantized),
    gemm_token_1_quant(functional_1_dense_2_1_Cast_ReadVariableOp_0_quantized, functional_1_dense_2_1_Add_ReadVariableOp_0_quantized)
{}

AlphaGomoku::GodNet::RetType QuantizedNetwork::feed(const Tensor<uint8_t, 3, 15, 15>& input) {
    // Direct chaining approach - no intermediate tensor tracking
    
    // Unknown operation: QuantizeLinear
    functional_1_conv2d_1_BiasAdd_quant.feed(input);

    functional_1_batch_normalization_1_batchnorm_mul_1_quant.mul(functional_1_conv2d_1_BiasAdd_quant.output());
    functional_1_batch_normalization_1_batchnorm_add_1_quant.add(functional_1_conv2d_1_BiasAdd_quant.output());
    functional_1_conv2d_1_2_BiasAdd_quant.feed(functional_1_conv2d_1_BiasAdd_quant.output());

    functional_1_batch_normalization_1_2_batchnorm_mul_1_quant.mul(functional_1_conv2d_1_2_BiasAdd_quant.output());
    functional_1_batch_normalization_1_2_batchnorm_add_1_quant.add(functional_1_conv2d_1_2_BiasAdd_quant.output());
    functional_1_conv2d_2_1_BiasAdd_quant.feed(functional_1_conv2d_1_2_BiasAdd_quant.output());

    functional_1_batch_normalization_2_1_batchnorm_mul_1_quant.mul(functional_1_conv2d_2_1_BiasAdd_quant.output());
    functional_1_batch_normalization_2_1_batchnorm_add_1_quant.add(functional_1_conv2d_2_1_BiasAdd_quant.output());
    QElemWise</* a_scale */ functional_1_batch_normalization_2_1_batchnorm_add_1_0_scale, /* a_zp */ functional_1_batch_normalization_2_1_batchnorm_add_1_0_zero_point, /* b_scale */ functional_1_batch_normalization_1_batchnorm_add_1_0_scale, /* b_zp */ functional_1_batch_normalization_1_batchnorm_add_1_0_zero_point, /* c_scale */ functional_1_add_1_Add_0_scale, /* c_zp */ functional_1_add_1_Add_0_zero_point, /* C */ 32, /* H */ 15, /* W */ 15>::add(functional_1_conv2d_2_1_BiasAdd_quant.output(), functional_1_conv2d_1_BiasAdd_quant.output());
    functional_1_conv2d_3_1_BiasAdd_quant.feed(functional_1_conv2d_2_1_BiasAdd_quant.output());

    functional_1_batch_normalization_3_1_batchnorm_mul_1_quant.mul(functional_1_conv2d_3_1_BiasAdd_quant.output());
    functional_1_batch_normalization_3_1_batchnorm_add_1_quant.add(functional_1_conv2d_3_1_BiasAdd_quant.output());
    functional_1_conv2d_4_1_BiasAdd_quant.feed(functional_1_conv2d_3_1_BiasAdd_quant.output());

    functional_1_batch_normalization_4_1_batchnorm_mul_1_quant.mul(functional_1_conv2d_4_1_BiasAdd_quant.output());
    functional_1_batch_normalization_4_1_batchnorm_add_1_quant.add(functional_1_conv2d_4_1_BiasAdd_quant.output());
    QElemWise</* a_scale */ functional_1_batch_normalization_4_1_batchnorm_add_1_0_scale, /* a_zp */ functional_1_batch_normalization_4_1_batchnorm_add_1_0_zero_point, /* b_scale */ functional_1_add_1_Add_0_scale, /* b_zp */ functional_1_add_1_Add_0_zero_point, /* c_scale */ functional_1_add_1_2_Add_0_scale, /* c_zp */ functional_1_add_1_2_Add_0_zero_point, /* C */ 32, /* H */ 15, /* W */ 15>::add(functional_1_conv2d_4_1_BiasAdd_quant.output(), functional_1_conv2d_2_1_BiasAdd_quant.output());
    functional_1_conv2d_5_1_BiasAdd_quant.feed(functional_1_conv2d_4_1_BiasAdd_quant.output());

    functional_1_batch_normalization_5_1_batchnorm_mul_1_quant.mul(functional_1_conv2d_5_1_BiasAdd_quant.output());
    functional_1_batch_normalization_5_1_batchnorm_add_1_quant.add(functional_1_conv2d_5_1_BiasAdd_quant.output());
    functional_1_conv2d_6_1_BiasAdd_quant.feed(functional_1_conv2d_5_1_BiasAdd_quant.output());

    functional_1_batch_normalization_6_1_batchnorm_mul_1_quant.mul(functional_1_conv2d_6_1_BiasAdd_quant.output());
    functional_1_batch_normalization_6_1_batchnorm_add_1_quant.add(functional_1_conv2d_6_1_BiasAdd_quant.output());
    QElemWise</* a_scale */ functional_1_batch_normalization_6_1_batchnorm_add_1_0_scale, /* a_zp */ functional_1_batch_normalization_6_1_batchnorm_add_1_0_zero_point, /* b_scale */ functional_1_add_1_2_Add_0_scale, /* b_zp */ functional_1_add_1_2_Add_0_zero_point, /* c_scale */ functional_1_add_2_1_Add_0_scale, /* c_zp */ functional_1_add_2_1_Add_0_zero_point, /* C */ 32, /* H */ 15, /* W */ 15>::add(functional_1_conv2d_6_1_BiasAdd_quant.output(), functional_1_conv2d_4_1_BiasAdd_quant.output());
    functional_1_conv2d_7_1_BiasAdd_quant.feed(functional_1_conv2d_6_1_BiasAdd_quant.output());

    functional_1_conv2d_8_1_convolution_quant.feed(functional_1_conv2d_6_1_BiasAdd_quant.output());

    functional_1_batch_normalization_7_1_batchnorm_mul_1_quant.mul(functional_1_conv2d_7_1_BiasAdd_quant.output());
    functional_1_batch_normalization_8_1_batchnorm_mul_1_quant.mul(functional_1_conv2d_8_1_convolution_quant.output());
    functional_1_batch_normalization_7_1_batchnorm_add_1_quant.add(functional_1_conv2d_7_1_BiasAdd_quant.output());
    functional_1_batch_normalization_8_1_batchnorm_add_1_quant.add(functional_1_conv2d_8_1_convolution_quant.output());
    functional_1_conv2d_7_1_BiasAdd_quant.output().flatten(); // Reshape: functional_1/flatten_2/Reshape
    functional_1_conv2d_8_1_convolution_quant.output().flatten(); // Reshape: functional_1/flatten_1_1/Reshape
    gemm_quant.feed(functional_1_conv2d_7_1_BiasAdd_quant.output().flatten());

    gemm_token_0_quant.feed(functional_1_conv2d_8_1_convolution_quant.output().flatten());

    // Unknown operation: DequantizeLinear
    gemm_token_1_quant.feed(gemm_token_0_quant.output());

    pi = softmax(gemm_quant.output()); // Softmax activation
    // Unknown operation: DequantizeLinear
    v = Tanh::call(gemm_token_1_quant.output()[0]); // Tanh activation

    return output();
}
} // namespace Black

} // namespace AlphaGomoku
  