

#ifndef OUT_WEIGHTS_H_
#define OUT_WEIGHTS_H_

#include <stdint.h>
#include "tensor1.h"
extern const char *fc1_tile_dtype; 
extern const char *fc1_alphas_0_dtype; 
extern const char *fc1_alphas_1_dtype; 
extern const char *fc1_alphas_2_dtype; 
extern const char *fc1_alphas_3_dtype; 
extern const char *fc2_tile_dtype; 
extern const char *fc2_alphas_0_dtype; 
extern const char fc1_tile_dim;
extern const char fc1_alphas_0_dim;
extern const char fc1_alphas_1_dim;
extern const char fc1_alphas_2_dim;
extern const char fc1_alphas_3_dim;
extern const char fc2_tile_dim;
extern const char fc2_alphas_0_dim;
extern const uint32_t fc1_tile_shape[] ;
extern const uint32_t fc1_alphas_0_shape[] ;
extern const uint32_t fc1_alphas_1_shape[] ;
extern const uint32_t fc1_alphas_2_shape[] ;
extern const uint32_t fc1_alphas_3_shape[] ;
extern const uint32_t fc2_tile_shape[] ;
extern const uint32_t fc2_alphas_0_shape[] ;
extern const uint64_t fc1_tile_size ;
extern const uint64_t fc1_alphas_0_size ;
extern const uint64_t fc1_alphas_1_size ;
extern const uint64_t fc1_alphas_2_size ;
extern const uint64_t fc1_alphas_3_size ;
extern const uint64_t fc2_tile_size ;
extern const uint64_t fc2_alphas_0_size ;
extern const uint8_t fc1_tile_data[]; 
extern const float fc1_alphas_0_data;; 
extern const float fc1_alphas_1_data; 
extern const float fc1_alphas_2_data; 
extern const float fc1_alphas_3_data; 
extern const uint8_t fc2_tile_data[]; 
extern const float fc2_alphas_0_data; 

//extern const unsigned char model_array[];
//extern const int model_array_len;

#endif  // OUT_WEIGHTS_H_
    