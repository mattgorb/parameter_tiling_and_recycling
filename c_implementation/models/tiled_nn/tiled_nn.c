



#include "tiled_nn.h"
#include "tensor1.h"
#include "util.h"
#include "kernels.h"
#include "layers.h"
#include "tiled_nn_weights.h"

TensorFloat1D* tiled_nn_forward(TensorFloat1D** test_img){
    TensorFloat1D* fc1_out=fc1(test_img);
    TensorFloat1D* fc2_out=fc2(&fc1_out);
    return fc2_out;
}



TensorFloat1D* fc1(TensorFloat1D** input){
    struct TiledFC layer_tile;

    layer_tile.tile=fc1_tile_data;
    layer_tile.tile_length=25088;
    layer_tile.alphas[0]=fc1_alphas_0_data;
    layer_tile.alphas[1]=fc1_alphas_1_data;
    layer_tile.alphas[2]=fc1_alphas_2_data;
    layer_tile.alphas[3]=fc1_alphas_3_data;


    bool fuse_relu=true;

    struct TiledFC* layer_tr = &layer_tile;
    
    uint8_t output_size=128;
    TensorFloat1D* output=create_empty_float_tensor_1d(output_size);

    uint16_t rows=128;
    uint16_t cols=784;



    output = tiled_fc(layer_tr, input,  output, fuse_relu, rows, cols );
    destroy_ptr((void**)input);

    return output;
}

TensorFloat1D* fc2(TensorFloat1D** input){
    
    struct TiledFC layer_tile;

    layer_tile.tile=fc2_tile_data;
    layer_tile.tile_length=1280;
    layer_tile.alphas[0]=fc2_alphas_0_data;

    bool fuse_relu=false;

    struct TiledFC* layer_tr = &layer_tile;
    
    uint8_t output_size=10;
    
    TensorFloat1D* output=create_empty_float_tensor_1d(output_size);

    uint16_t rows=10;
    uint16_t cols=128;

    
    output = tiled_fc(layer_tr, input,  output, fuse_relu, rows, cols );
    destroy_ptr((void**)input);

    return output;
}