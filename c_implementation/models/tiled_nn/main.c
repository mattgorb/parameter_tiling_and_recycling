#include <stdio.h>
#include "tensor1.h"
#include "util.h"
#include "tiled_nn_weights.h"
#include "tiled_nn.h"

int main() {

    
    TensorFloat1D* test_img=tensor_test_image_mnist();

    TensorFloat1D* output=tiled_nn_forward(&test_img);


    /*printf("Output: ");
    for(int i=0;i<10;i++){
        printf("%4f, ",output->data[i]);
    }
    printf("\n");
    */

    return 0;
}
