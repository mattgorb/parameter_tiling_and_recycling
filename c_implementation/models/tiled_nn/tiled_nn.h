#ifndef SBNN_FP_H
#define SBNN_FP_H


#include <stdio.h>
#include "tensor1.h"


TensorFloat1D* tiled_nn_forward(TensorFloat1D** input);
TensorFloat1D* fc1(TensorFloat1D** input);
TensorFloat1D* fc2(TensorFloat1D** input);
#endif 