import ttnn
import torch
import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch.utils._pytree as pytree
from torch._dynamo.output_graph import FakeRootModule
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
)

ttnn_ROW_MAJOR_LAYOUT = ttnn.ROW_MAJOR_LAYOUT
ttnn_TILE_LAYOUT = ttnn.TILE_LAYOUT
ttnn_bfloat16 = ttnn.bfloat16
ttnn_uint32 = ttnn.uint32

def project_qkv(ttnn_Specified_Device, input_tensor, weight, bias):
    """Projects input tensor using weight matrix and bias.
    
    Args:
        ttnn_Specified_Device: Device to run operations on
        input_tensor: Input tensor to project (256, 1024)
        weight: Weight matrix for projection
        bias: Bias tensor to add after projection
        
    Returns:
        Projected tensor of shape (1, 256, 1024)
    """
    ttnn_from_torch = ttnn.from_torch(weight, device=ttnn_Specified_Device, layout=ttnn_TILE_LAYOUT, dtype=ttnn_bfloat16)
    ttnn_transpose = ttnn.transpose(ttnn_from_torch, 0, 1)
    ttnn_matmul = ttnn.matmul(input_tensor, ttnn_transpose)
    ttnn_from_torch_bias = ttnn.from_torch(bias, device=ttnn_Specified_Device, layout=ttnn_TILE_LAYOUT, dtype=ttnn_bfloat16)
    ttnn_add = ttnn.add(ttnn_from_torch_bias, ttnn_matmul)
    return ttnn.reshape(ttnn_add, (1, 256, 1024))

def process_layer(ttnn_Specified_Device, ttnn_reshape_2, ttnn_layer_norm, ttnn_multiply,
                 arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1,
                 arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1,
                 arg17_1, arg18_1, arg19_1, arg20_1):
    """A single layer of BERT Large model.
    
    Architecture details:
    - Hidden size: 1024
    - Number of attention heads: 16
    - Head dimension: 64 (1024/16)
    - Intermediate size: 4096 (4x hidden size)
    - Sequence length: 256
    
    The layer consists of:
    1. Multi-Head Self-Attention Block
       - Projects input into Query, Key, Value matrices
       - Splits into 16 attention heads
       - Computes scaled dot-product attention
       - Concatenates heads and projects back
    
    2. Add & Norm Block
       - Residual connection with input
       - Layer normalization
    
    3. Feed Forward Block
       - Two linear transformations with GELU activation
       - Expands to intermediate size (4096) then back to hidden size (1024)
    
    4. Final Add & Norm Block
       - Residual connection
       - Layer normalization
    """
    
    # ---------- Multi-Head Self-Attention Block ----------
    # Project input into Query, Key, Value matrices (256, 1024)
    ttnn_reshape_3 = project_qkv(ttnn_Specified_Device, ttnn_reshape_2, arg5_1, arg6_1)  # Query projection
    ttnn_reshape_5 = project_qkv(ttnn_Specified_Device, ttnn_reshape_2, arg7_1, arg8_1)  # Key projection
    ttnn_reshape_8 = project_qkv(ttnn_Specified_Device, ttnn_reshape_2, arg9_1, arg10_1)  # Value projection

    # Split into attention heads (1, 256, 16, 64)
    ttnn_reshape_6 = ttnn.reshape(ttnn_reshape_5, (1, 256, 16, 64))
    ttnn_permute = ttnn.permute(ttnn_reshape_6, (0, 2, 1, 3))
    ttnn_reshape_9 = ttnn.reshape(ttnn_reshape_8, (1, 256, 16, 64))
    ttnn_permute_1 = ttnn.permute(ttnn_reshape_9, (0, 2, 1, 3))
    ttnn_reshape_10 = ttnn.reshape(ttnn_reshape_3, (1, 256, 16, 64))
    ttnn_permute_2 = ttnn.permute(ttnn_reshape_10, (0, 2, 1, 3))

    # Compute scaled dot-product attention
    ttnn_transpose_3 = ttnn.transpose(ttnn_permute, 3, 2)
    ttnn_reshape_11 = ttnn.reshape(ttnn_permute_2, (16, 256, 64))
    ttnn_reshape_12 = ttnn.reshape(ttnn_transpose_3, (16, 64, 256))
    ttnn_matmul_3 = ttnn.matmul(ttnn_reshape_11, ttnn_reshape_12)
    ttnn_reshape_13 = ttnn.reshape(ttnn_matmul_3, (1, 16, 256, 256))

    # Scale attention scores and apply attention mask
    ttnn_multiply_1 = ttnn.multiply(ttnn_reshape_13, 0.125)  # Scale by 1/sqrt(head_dim)
    ttnn_add_147 = ttnn.add(ttnn_multiply_1, ttnn_multiply)
    ttnn_softmax = ttnn.softmax(ttnn_add_147, -1, numeric_stable = True)

    # Apply attention to values
    ttnn_reshape_14 = ttnn.reshape(ttnn_softmax, (16, 256, 256))
    ttnn_reshape_15 = ttnn.reshape(ttnn_permute_1, (16, 256, 64))
    ttnn_matmul_4 = ttnn.matmul(ttnn_reshape_14, ttnn_reshape_15)

    # Reshape and transpose back to original dimensions
    ttnn_reshape_16 = ttnn.reshape(ttnn_matmul_4, (1, 16, 256, 64))
    ttnn_permute_3 = ttnn.permute(ttnn_reshape_16, (0, 2, 1, 3))
    ttnn_reshape_17 = ttnn.reshape(ttnn_permute_3, (1, 256, 1024))
    ttnn_reshape_18 = ttnn.reshape(ttnn_reshape_17, (256, 1024))

    # ---------- First Add & Norm Block ----------
    # Project attention output
    ttnn_from_torch_15 = ttnn.from_torch(arg11_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_transpose_4 = ttnn.transpose(ttnn_from_torch_15, 0, 1)
    ttnn_matmul_5 = ttnn.matmul(ttnn_reshape_18, ttnn_transpose_4)
    ttnn_from_torch_16 = ttnn.from_torch(arg12_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_add_3 = ttnn.add(ttnn_from_torch_16, ttnn_matmul_5)
    ttnn_reshape_19 = ttnn.reshape(ttnn_add_3, (1, 256, 1024))

    # Add residual connection and apply layer normalization
    ttnn_add_148 = ttnn.add(ttnn_reshape_19, ttnn_layer_norm)
    ttnn_from_torch_17 = ttnn.from_torch(arg13_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_from_torch_18 = ttnn.from_torch(arg14_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_layer_norm_1 = ttnn.layer_norm(ttnn_add_148, epsilon = 1e-12, weight = ttnn_from_torch_17, bias = ttnn_from_torch_18)
    ttnn_reshape_20 = ttnn.reshape(ttnn_layer_norm_1, (256, 1024))

    # ---------- Feed Forward Block ----------
    # First linear transformation (expand to 4x hidden size)
    ttnn_from_torch_19 = ttnn.from_torch(arg15_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_transpose_5 = ttnn.transpose(ttnn_from_torch_19, 0, 1)
    ttnn_matmul_6 = ttnn.matmul(ttnn_reshape_20, ttnn_transpose_5)
    ttnn_from_torch_20 = ttnn.from_torch(arg16_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_add_4 = ttnn.add(ttnn_from_torch_20, ttnn_matmul_6)
    ttnn_reshape_21 = ttnn.reshape(ttnn_add_4, (1, 256, 4096))

    # Apply GELU activation
    ttnn_gelu = ttnn.gelu(ttnn_reshape_21)
    ttnn_reshape_22 = ttnn.reshape(ttnn_gelu, (256, 4096))

    # Second linear transformation (project back to hidden size)
    ttnn_from_torch_21 = ttnn.from_torch(arg17_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_transpose_6 = ttnn.transpose(ttnn_from_torch_21, 0, 1)
    ttnn_matmul_7 = ttnn.matmul(ttnn_reshape_22, ttnn_transpose_6)
    ttnn_from_torch_22 = ttnn.from_torch(arg18_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_add_5 = ttnn.add(ttnn_from_torch_22, ttnn_matmul_7)
    ttnn_reshape_23 = ttnn.reshape(ttnn_add_5, (1, 256, 1024))

    # ---------- Second Add & Norm Block ----------
    # Add residual connection
    ttnn_add_149 = ttnn.add(ttnn_reshape_23, ttnn_layer_norm_1)

    # Apply layer normalization
    ttnn_from_torch_23 = ttnn.from_torch(arg19_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_from_torch_24 = ttnn.from_torch(arg20_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_layer_norm_2 = ttnn.layer_norm(ttnn_add_149, epsilon = 1e-12, weight = ttnn_from_torch_23, bias = ttnn_from_torch_24)
    ttnn_reshape_24 = ttnn.reshape(ttnn_layer_norm_2, (256, 1024))
    
    return ttnn_reshape_24, ttnn_layer_norm_2


def forward(ttnn_Specified_Device, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1):
    ttnn_ROW_MAJOR_LAYOUT = ttnn.ROW_MAJOR_LAYOUT
    ttnn_TILE_LAYOUT = ttnn.TILE_LAYOUT
    ttnn_bfloat16 = ttnn.bfloat16
    ttnn_uint32 = ttnn.uint32

    # ------------- INPUT ------------------
    ttnn_from_torch = ttnn.from_torch(arg393_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_uint32)
    ttnn_reshape = ttnn.reshape(ttnn_from_torch, (1, 1, 256))
    ttnn_reshape_1 = ttnn.reshape(ttnn_reshape, (1, 1, 1, 256))
    ttnn_typecast = ttnn.typecast(ttnn_reshape_1, ttnn_bfloat16)
    ttnn_rsub = ttnn.rsub(ttnn_typecast, 1.0)
    ttnn_multiply = ttnn.multiply(ttnn_rsub, -3.3895313892515355e+38)
    ttnn_from_torch_1 = ttnn.from_torch(arg391_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32)
    ttnn_slice = ttnn.slice(ttnn_from_torch_1, [0, 0], [1, 256])
    ttnn_from_torch_2 = ttnn.from_torch(arg392_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32)
    ttnn_from_torch_3 = ttnn.from_torch(arg0_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_embedding = ttnn.embedding(ttnn_from_torch_2, ttnn_from_torch_3, layout = ttnn_TILE_LAYOUT)
    ttnn_from_torch_4 = ttnn.from_torch(arg394_1, device = ttnn_Specified_Device, layout = ttnn_ROW_MAJOR_LAYOUT, dtype = ttnn_uint32)
    ttnn_from_torch_5 = ttnn.from_torch(arg1_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_embedding_1 = ttnn.embedding(ttnn_from_torch_4, ttnn_from_torch_5, layout = ttnn_TILE_LAYOUT)
    ttnn_add_145 = ttnn.add(ttnn_embedding, ttnn_embedding_1)
    ttnn_to_layout = ttnn.to_layout(ttnn_slice, ttnn_ROW_MAJOR_LAYOUT)
    ttnn_from_torch_6 = ttnn.from_torch(arg2_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_embedding_2 = ttnn.embedding(ttnn_to_layout, ttnn_from_torch_6, layout = ttnn_TILE_LAYOUT)
    ttnn_add_146 = ttnn.add(ttnn_add_145, ttnn_embedding_2)
    ttnn_from_torch_7 = ttnn.from_torch(arg3_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_from_torch_8 = ttnn.from_torch(arg4_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_layer_norm = ttnn.layer_norm(ttnn_add_146, epsilon = 1e-12, weight = ttnn_from_torch_7, bias = ttnn_from_torch_8)
    ttnn_reshape_2 = ttnn.reshape(ttnn_layer_norm, (256, 1024))

    # ------------ LAYER 0 ------------------
    ttnn_reshape_24, ttnn_layer_norm_2 = process_layer(ttnn_Specified_Device, ttnn_reshape_2, ttnn_layer_norm, ttnn_multiply, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1)
    # -------------- LAYER 1 ------------------
    ttnn_reshape_46, ttnn_layer_norm_4 = process_layer(ttnn_Specified_Device, ttnn_reshape_24, ttnn_layer_norm_2, ttnn_multiply, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1)
    
    # -------------- LAYER 2 ------------------
    ttnn_reshape_68, ttnn_layer_norm_6 = process_layer(ttnn_Specified_Device, ttnn_reshape_46, ttnn_layer_norm_4, ttnn_multiply, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1)
    
    # -------------- LAYER 3 ------------------
    ttnn_reshape_90, ttnn_layer_norm_8 = process_layer(ttnn_Specified_Device, ttnn_reshape_68, ttnn_layer_norm_6, ttnn_multiply, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1)
    
    # -------------- LAYER 4 ------------------
    ttnn_reshape_112, ttnn_layer_norm_10 = process_layer(ttnn_Specified_Device, ttnn_reshape_90, ttnn_layer_norm_8, ttnn_multiply, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1)
    # -------------- LAYER 5 ------------------
    ttnn_reshape_134, ttnn_layer_norm_12 = process_layer(ttnn_Specified_Device, ttnn_reshape_112, ttnn_layer_norm_10, ttnn_multiply, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1)
    # -------------- LAYER 6 ------------------
    ttnn_reshape_156, ttnn_layer_norm_14 = process_layer(ttnn_Specified_Device, ttnn_reshape_134, ttnn_layer_norm_12, ttnn_multiply, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1)
    # -------------- LAYER 7 ------------------
    ttnn_reshape_178, ttnn_layer_norm_16 = process_layer(ttnn_Specified_Device, ttnn_reshape_156, ttnn_layer_norm_14, ttnn_multiply, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1)
    # -------------- LAYER 8 ------------------
    ttnn_reshape_200, ttnn_layer_norm_18 = process_layer(ttnn_Specified_Device, ttnn_reshape_178, ttnn_layer_norm_16, ttnn_multiply, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1)

    # -------------- LAYER 9 -----------------
    ttnn_reshape_222, ttnn_layer_norm_20 = process_layer(ttnn_Specified_Device, ttnn_reshape_200, ttnn_layer_norm_18, ttnn_multiply, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1)

    # -------------- LAYER 10 ----------------
    ttnn_reshape_244, ttnn_layer_norm_22 = process_layer(ttnn_Specified_Device, ttnn_reshape_222, ttnn_layer_norm_20, ttnn_multiply, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1)

    # -------------- LAYER 11 ----------------

    ttnn_reshape_266, ttnn_layer_norm_24 = process_layer(ttnn_Specified_Device, ttnn_reshape_244, ttnn_layer_norm_22, ttnn_multiply, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1)

    # -------------- LAYER 12 ----------------
    ttnn_reshape_288, ttnn_layer_norm_26 = process_layer(ttnn_Specified_Device, ttnn_reshape_266, ttnn_layer_norm_24, ttnn_multiply, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1)


    # -------------- LAYER 13 ----------------
    ttnn_reshape_310, ttnn_layer_norm_28 = process_layer(ttnn_Specified_Device, ttnn_reshape_288, ttnn_layer_norm_26, ttnn_multiply, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1)

    # -------------- LAYER 14 ----------------
    ttnn_reshape_332, ttnn_layer_norm_30 = process_layer(ttnn_Specified_Device, ttnn_reshape_310, ttnn_layer_norm_28, ttnn_multiply, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1)

    # -------------- LAYER 15 ----------------
    ttnn_reshape_354, ttnn_layer_norm_32 = process_layer(ttnn_Specified_Device, ttnn_reshape_332, ttnn_layer_norm_30, ttnn_multiply, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1)

    # -------------- LAYER 16 ----------------
    ttnn_reshape_376, ttnn_layer_norm_34 = process_layer(ttnn_Specified_Device, ttnn_reshape_354, ttnn_layer_norm_32, ttnn_multiply, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1)

    # -------------- LAYER 17 ----------------
    ttnn_reshape_398, ttnn_layer_norm_36 = process_layer(ttnn_Specified_Device, ttnn_reshape_376, ttnn_layer_norm_34, ttnn_multiply, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1)

    # -------------- LAYER 18 ----------------
    ttnn_reshape_420, ttnn_layer_norm_38 = process_layer(ttnn_Specified_Device, ttnn_reshape_398, ttnn_layer_norm_36, ttnn_multiply, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1)

    
    # -------------- LAYER 19 ----------------
    ttnn_reshape_442, ttnn_layer_norm_40 = process_layer(ttnn_Specified_Device, ttnn_reshape_420, ttnn_layer_norm_38, ttnn_multiply, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1)
    
    # -------------- LAYER 20 ----------------
    
    ttnn_reshape_464, ttnn_layer_norm_42 = process_layer(ttnn_Specified_Device, ttnn_reshape_442, ttnn_layer_norm_40, ttnn_multiply, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1)

    # -------------- LAYER 21 ----------------
    ttnn_reshape_486, ttnn_layer_norm_44 = process_layer(ttnn_Specified_Device, ttnn_reshape_464, ttnn_layer_norm_42, ttnn_multiply, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1)

    # -------------- LAYER 22 ----------------

    ttnn_reshape_508, ttnn_layer_norm_46 = process_layer(ttnn_Specified_Device, ttnn_reshape_486, ttnn_layer_norm_44, ttnn_multiply, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1)

    # -------------- LAYER 2 ----------------
    ttnn_reshape_530, ttnn_layer_norm_48 = process_layer(ttnn_Specified_Device, ttnn_reshape_508, ttnn_layer_norm_46, ttnn_multiply, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1)
    # -------------- OUTPUT ----------------
    
    ttnn_from_torch_393 = ttnn.from_torch(arg389_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_transpose_168 = ttnn.transpose(ttnn_from_torch_393, 0, 1)
    ttnn_matmul_192 = ttnn.matmul(ttnn_reshape_530, ttnn_transpose_168)
    ttnn_from_torch_394 = ttnn.from_torch(arg390_1, device = ttnn_Specified_Device, layout = ttnn_TILE_LAYOUT, dtype = ttnn_bfloat16)
    ttnn_add_144 = ttnn.add(ttnn_from_torch_394, ttnn_matmul_192)
    ttnn_reshape_531 = ttnn.reshape(ttnn_add_144, (1, 256, 2))
    ttnn_to_layout_1 = ttnn.to_layout(ttnn_reshape_531, ttnn_ROW_MAJOR_LAYOUT)
    ttnn_split = ttnn.split(ttnn_to_layout_1, 1, 2)
    getitem_49 = ttnn_split[0]
    getitem_50 = ttnn_split[1]
    ttnn_to_layout_2 = ttnn.to_layout(getitem_49, ttnn_TILE_LAYOUT)
    ttnn_reshape_532 = ttnn.reshape(ttnn_to_layout_2, (1, 256))
    ttnn_to_layout_3 = ttnn.to_layout(getitem_50, ttnn_TILE_LAYOUT)
    ttnn_reshape_533 = ttnn.reshape(ttnn_to_layout_3, (1, 256))
    ttnn_to_torch = ttnn.to_torch(ttnn_reshape_532, dtype = torch.bfloat16)
    ttnn_to_torch_1 = ttnn.to_torch(ttnn_reshape_533, dtype = torch.bfloat16)
    return (ttnn_to_torch, ttnn_to_torch_1)


def run_bert(ttnn_Specified_Device, iteration):
     # Download model from cloud
    model_name = "phiyodr/bert-large-finetuned-squad2"
    m = AutoModelForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", torch_dtype=torch.bfloat16)
    context = 'Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. "The prophet and founding hero of modern archaeology", Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art. '
    question = "What discipline did Winkelmann create?"
    inputs = tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        )
    
    graph = torch.fx.Graph()
    modules = {}
    modules['self'] = m
    root = FakeRootModule(modules)
    gm = torch.fx.GraphModule(root, graph)

    full_args = []
    params = {
        **dict(m.named_parameters(remove_duplicate=False)),
        **dict(m.named_buffers(remove_duplicate=False)),
    }

    params_items = list(params.items())
    params_items[1], params_items[2] = params_items[2], params_items[1]
    params_items.pop()
    params = dict(params_items)
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = list(params_flat)
    full_args.extend(params_flat)
    profiler.enable()
    profiler.start(iteration)
    forward(ttnn_Specified_Device, *full_args, inputs.data['input_ids'], inputs.data['token_type_ids'], inputs.data['attention_mask'])
    profiler.end(iteration)
        
if __name__ == "__main__":
    device_id = 0
    ttnn_Specified_Device = ttnn.open_device(device_id=device_id)
    ttnn.disable_and_clear_program_cache(ttnn_Specified_Device)
    disable_persistent_kernel_cache()
    run_bert(ttnn_Specified_Device, 0)
    
    enable_persistent_kernel_cache()
    ttnn.enable_program_cache(ttnn_Specified_Device)
    
    for i in range(1, 5):
        run_bert(ttnn_Specified_Device, i)

    ttnn.close_device(ttnn_Specified_Device)
    for i in range(5):
        print(f"Runtime Iteration {i}: {profiler.get(i)}")