#include "tensor1.h"
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "util.h"






TensorFloat1D create_tensor_float_1d(const char* dtype,  const char dim, const uint32_t shape[],const uint8_t data[], const uint64_t size){
    t1_assert(get_entry_data_type(dtype)==0,"data type not equal to float32");
    t1_assert(dim==1,"dim is not equal to 1");

    // Allocate memory for the Tensor structure
    TensorFloat1D tensor;// = (TensorFloat2D*)malloc(sizeof(TensorFloat2D));

    // Initialize the fields of the tensor
    tensor.shape = shape[0];
    tensor.size = (int64_t)size;
    
    // Allocate memory for the matrix
    tensor.data = (float*)malloc(tensor.shape * sizeof(float*));

    float* tensor_data = (float*)data;
    for (uint32_t i = 0; i < tensor.shape; i++) {
        tensor.data[i]=tensor_data[i];
    }
    return tensor;
}



TensorFloat2D create_tensor_float_2d(const char* dtype, 
     const char dim,
     const uint32_t shape[],
     const uint8_t data[],
     const uint64_t size){

    
    t1_assert(get_entry_data_type(dtype)==0,"data type not equal to float32");
    t1_assert(dim==2,"dim is not equal to 2");

    // Allocate memory for the Tensor structure
    TensorFloat2D tensor;// = (TensorFloat2D*)malloc(sizeof(TensorFloat2D));

    // Initialize the fields of the tensor
    tensor.shape = (uint32_t*)shape;
    tensor.size = (int64_t)size;
    tensor.rows = tensor.shape[0];
    tensor.cols = tensor.shape[1];

    // Allocate memory for the matrix
    tensor.data = (float**)malloc(tensor.rows * sizeof(float*));

    float* tensor_data = (float*)data;
    for (uint32_t i = 0; i < tensor.rows; i++) {
        tensor.data[i] = (float*)malloc(tensor.cols * sizeof(float));
        for (uint32_t j = 0; j < tensor.cols; j++) {
            tensor.data[i][j] = tensor_data[i * tensor.cols + j];
        }
    }
    return tensor;
}






TensorUInt82D create_tensor_uint8_2d(const char* dtype, 
     const char dim,
     const uint32_t shape[],
     const uint8_t data[],
     const uint64_t size){

    
    t1_assert(get_entry_data_type(dtype)==1,"data type not equal to uint8");
    t1_assert(dim==2,"dim is not equal to 2");

    // Allocate memory for the Tensor structure
    TensorUInt82D tensor;// = (TensorFloat2D*)malloc(sizeof(TensorFloat2D));

    // Initialize the fields of the tensor
    tensor.shape = (uint32_t*)shape;
    tensor.size = (int64_t)size;
    tensor.rows = tensor.shape[0];
    tensor.cols = tensor.shape[1];

    // Allocate memory for the matrix
    tensor.data = (uint8_t**)malloc(tensor.rows * sizeof(uint8_t*));

    uint8_t* tensor_data = (uint8_t*)data;
    for (uint32_t i = 0; i < tensor.rows; i++) {
        tensor.data[i] = (uint8_t*)malloc(tensor.cols * sizeof(uint8_t));
        for (uint32_t j = 0; j < tensor.cols; j++) {
            tensor.data[i][j] = tensor_data[i * tensor.cols + j];
        }
    }
    return tensor;
}



TensorInt82D create_tensor_int8_2d(const char* dtype, 
     const char dim,
     const uint32_t shape[],
     const uint8_t data[],
     const uint64_t size){

    
    t1_assert(get_entry_data_type(dtype)==1,"data type not equal to uint8");
    t1_assert(dim==2,"dim is not equal to 2");

    // Allocate memory for the Tensor structure
    TensorInt82D tensor;// = (TensorFloat2D*)malloc(sizeof(TensorFloat2D));

    // Initialize the fields of the tensor
    tensor.shape = (uint32_t*)shape;
    tensor.size = (int64_t)size;
    tensor.rows = tensor.shape[0];
    tensor.cols = tensor.shape[1];

    // Allocate memory for the matrix
    tensor.data = (int8_t**)malloc(tensor.rows * sizeof(int8_t*));

    int8_t* tensor_data = (int8_t*)data;
    for (uint32_t i = 0; i < tensor.rows; i++) {
        tensor.data[i] = (int8_t*)malloc(tensor.cols * sizeof(int8_t));
        for (uint32_t j = 0; j < tensor.cols; j++) {
            tensor.data[i][j] = tensor_data[i * tensor.cols + j];
            
        }
    }
    return tensor;
}







TensorFloat1D* create_empty_float_tensor_1d(uint32_t numElements)
{
    TensorFloat1D* tensor = (TensorFloat1D*)malloc(sizeof(TensorFloat1D));

    // Allocate memory for the array of float values
    float *empty = (float *)malloc(numElements * sizeof(float));
    tensor->size = sizeof(empty);
    tensor->data = empty;
    tensor->shape=numElements;
    return tensor;
}

TensorInt81D* create_empty_int8_tensor_1d(uint32_t numElements)
{
    TensorInt81D* tensor = (TensorInt81D*)malloc(sizeof(TensorInt81D));

    // Allocate memory for the array of float values
    int8_t *empty = (int8_t *)malloc(numElements * sizeof(int8_t));
    tensor->size = sizeof(empty);
    tensor->data = empty;
    tensor->shape=numElements;
    return tensor;
}


TensorInt161D* create_empty_int16_tensor_1d(uint32_t numElements)
{
    TensorInt161D* tensor = (TensorInt161D*)malloc(sizeof(TensorInt161D));

    // Allocate memory for the array of float values
    int16_t *empty = (int16_t *)malloc(numElements * sizeof(int16_t));
    tensor->size = sizeof(empty);
    tensor->data = empty;
    tensor->shape=numElements;
    return tensor;
}



uint8_t get_entry_data_type(const char* entry_data_type){
    int result = strcmp(entry_data_type, "float32");
    if (result==0){
        return DATA_TYPE_FLOAT;
    }
    else{
        return DATA_TYPE_UINT8;
    }
}




/**/
void print_tensor(Tensor* tensor) {
    // Print tensor properties
    printf("ndim: %u\n", tensor->ndim);
    printf("size: %ld\n", tensor->size);
    printf("data_type: %u\n", tensor->data_type);
    
    // Print the first 5 values from the data array
    printf("data (first 5 values): ");
    for (int i = 0; i < 5; i++) {
        printf("%u ", tensor->data[i]);
    }
    printf("\n");

    // Print the shape
    printf("shape: [");
    for (int i = 0; i < tensor->ndim; i++) {
        printf("%u", tensor->shape[i]);
        if (i < tensor->ndim - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}








//Call like this: destroy_tensor_ptr(&tensor2);
void destroy_tensor_ptr(Tensor** x) {
    free(*x);
    *x = NULL;
}


void tensor_status(Tensor* x) {
    if (x == NULL) {
        printf("Pointer is null\n");
    } else {
        printf("Pointer is not null\n");
    }
}


void tensor_ptr_status(Tensor** x) {
    if (*x == NULL) {
        printf("Pointer is null\n");
    } else {
        printf("Pointer is not null\n");
    }
}











/**/
TensorFloat1D* uint_to_float32_scale(TensorUInt81D** test_img, float scale,double zero_point){


    // Allocate memory for the Tensor structure
    TensorFloat1D* tensor = (TensorFloat1D*)malloc(sizeof(TensorFloat1D));

    tensor->shape = (*test_img)->shape;
    tensor->size = (*test_img)->size;
    tensor->data = (float*)malloc((*test_img)->shape * sizeof(float));

    for (int i = 0; i < (*test_img)->shape; ++i) {
        tensor->data[i] =  ((float)((*test_img)->data[i] - (float)zero_point)*scale);
         //printf("%f\n", tensor->data[i]);
    }
    //exit(0);
    return tensor;
}
