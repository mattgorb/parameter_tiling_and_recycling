#ifndef TENSOR1_H
#define TENSOR1_H

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>



#define DATA_TYPE_FLOAT 0
#define DATA_TYPE_UINT8 1

//Generic Tensor
typedef struct {
    uint8_t ndim;
    uint32_t* shape;
    int64_t size;
    uint8_t data_type; // Indicates the data type (e.g., DATA_TYPE_FLOAT or DATA_TYPE_UINT8)
    uint8_t* data;    // Pointer to the tensor's data
} Tensor;

typedef struct {
    uint32_t* shape;
    uint32_t rows;
    uint32_t cols;
    int64_t size;
    float** data;    // Pointer to the tensor's data
} TensorFloat2D;

typedef struct {
    uint32_t shape;
    int64_t size;
    float* data;    // Pointer to the tensor's data
} TensorFloat1D;

typedef struct {
    uint32_t* shape;
    uint32_t rows;
    uint32_t cols;
    int64_t size;
    uint8_t** data;    // Pointer to the tensor's data
} TensorUInt82D;

typedef struct {
    uint32_t shape;
    int64_t size;
    uint8_t* data;    // Pointer to the tensor's data
} TensorUInt81D;

typedef struct {
    uint32_t shape;
    int64_t size;
    int8_t* data;    // Pointer to the tensor's data
} TensorInt81D;

typedef struct {
    uint32_t shape;
    int64_t size;
    int16_t* data;    // Pointer to the tensor's data
} TensorInt161D;

typedef struct {
    uint32_t* shape;
    uint32_t rows;
    uint32_t cols;
    int64_t size;
    int8_t** data;    // Pointer to the tensor's data
} TensorInt82D;

void print_tensor(Tensor* tensor);

void destroy_tensor(Tensor* tensor);
void destroy_tensor_ptr(Tensor** x);

void tensor_ptr_status(Tensor** x);
void tensor_status(Tensor* x);




uint8_t get_entry_data_type(const  char* entry_data_type);


TensorFloat1D* create_empty_float_tensor_1d(uint32_t numElements);
TensorInt81D* create_empty_int8_tensor_1d(uint32_t numElements);
TensorInt161D* create_empty_int16_tensor_1d(uint32_t numElements);


TensorFloat2D create_tensor_float_2d(const char* dtype,  const char dim, const uint32_t shape[],const uint8_t data[], const uint64_t size);
TensorFloat1D create_tensor_float_1d(const char* dtype,  const char dim, const uint32_t shape[],const uint8_t data[], const uint64_t size);
TensorUInt82D create_tensor_uint8_2d(const char* dtype,  const char dim, const uint32_t shape[],const uint8_t data[], const uint64_t size);
TensorInt82D  create_tensor_int8_2d(const char* dtype,  const char dim, const uint32_t shape[],const uint8_t data[], const uint64_t size);


TensorFloat1D* uint_to_float32_scale(TensorUInt81D** test_img, float scale, double zero_point);

#endif // TENSOR1_H
