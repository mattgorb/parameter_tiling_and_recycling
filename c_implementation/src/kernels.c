#include "kernels.h"
#include "util.h"




TensorFloat1D* tiled_fc(struct TiledFC* layer, TensorFloat1D** input, TensorFloat1D* output, bool fuse_relu, uint16_t rows, uint16_t cols) {
    uint32_t tile_ind=0;
    uint8_t tile_num=0;
    uint8_t k=7;
    for (int i = 0; i < rows; i++) {
        output->data[i] = 0.0f;
        for (int j = 0; j < cols; j++) {
            int8_t weight = (2*((layer->tile[tile_ind/8] >> k) & 1)-1); // Extract the k-th bit and convert from binary to +-1
            //int8_t weight = (((layer->tile[tile_ind/8] >> k) & 1)); // Extract the k-th bit and convert from binary to +-1

            output->data[i] += ((*input)->data[j]) * weight * layer->alphas[tile_num];
            
            tile_ind+=1;
            if(tile_ind>=layer->tile_length){
                //printf("%4f\n", layer->alphas[tile_num]);
                tile_ind=0;
                tile_num+=1;
            }
            if (k==0){
                k=7;
            }
            else{
                k-=1;
            }
        }

        if(fuse_relu==true){
            if(output->data[i]<0){
                output->data[i]=0;
            }
        }
        
    }
    return output;
}

