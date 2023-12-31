#ifndef KERNELS_H
#define KERNELS_H

#include <stdio.h>
#include "tensor1.h"
#include "layers.h"


TensorFloat1D* tiled_fc(struct TiledFC* layer, TensorFloat1D** input, TensorFloat1D* output, bool fuse_relu,uint16_t rows, uint16_t col);

#endif 