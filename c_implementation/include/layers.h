
#ifndef LAYERS_H
#define LAYERS_H

//DATA_TYPE_FLOAT 0
struct SBLinear_FP{
    TensorFloat2D weight;
    TensorFloat2D mask;
    float alpha;
    TensorFloat1D bias;
};


//DATA_TYPE_UINT8 1
struct TiledFC{
    //TensorUInt81D tile;
    const uint8_t* tile; 
    float alphas[4];
    uint32_t tile_length;  // Tile length field
};

//DATA_TYPE_UINT8 1
struct SBLinear{
    TensorUInt82D weight;
    TensorUInt82D mask;
    float alpha;
    TensorUInt81D bias;
};



//DATA_TYPE_UINT8 1
struct QInt8Linear{
    TensorInt82D weight;
    TensorFloat1D bias;
    double scale;
    double zero_point;

    double scale_weight;
    double zero_point_weight;
};


//DATA_TYPE_UINT8 1
struct Float32Linear{
    TensorFloat2D weight;
    TensorFloat1D bias;
};

#endif 