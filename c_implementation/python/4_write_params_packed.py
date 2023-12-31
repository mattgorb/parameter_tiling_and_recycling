import struct
import numpy as np

import argparse
import os
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import numpy as np





def write_blob(model_dict, filename):
    """
    Dumps parameter data

    Arguments:
        filename (string): filename for binary blob
    """

    filename = filename + '.bin'

    with open('../src/'+filename, "wb") as f:
        for w_name in model_dict.keys():
            print('Weight name:')
            print(w_name)

            w = model_dict[w_name]
            print(f'dtype: {w.dtype}')
            
            f.write(w_name.encode())
            f.write(bytes([0]))
            if w.dtype == np.float64:
                f.write(bytes(b'doubl'))
            if w.dtype == np.float32:

                f.write(bytes(b'float'))
            if w.dtype == np.int32:
                f.write(bytes(b'int32'))
            if w.dtype == np.int16:
                f.write(bytes(b'int16'))
            f.write(struct.pack("b", w.ndim))
            print(f'dim: {w.ndim}')
            for i in range(w.ndim):
                f.write(struct.pack("i", w.shape[i]))
            print(w)
            b = w.tobytes()
            f.write(struct.pack("Q", len(b)))
            f.write(struct.pack("Q", 0))
            f.write(b)




def write_c_blob(model_dict, filename):
    """
    Dumps parameter data

    Arguments:
        filename (string): filename for binary blob
    """

    #filename = filename + '.c'
    byte_array=[]

    name_ls=[]
    dtype_ls=[]
    dim_ls=[]
    shape_ls=[]
    byte_array=[]
    for w_name in model_dict.keys():

        w = model_dict[w_name]

        name_ls.append(w_name.replace('.','_'))
        dtype_ls.append(w.dtype)
        dim_ls.append(w.ndim)

        shape_ls.append(','.join([str(w.shape[i]) for i in range(w.ndim)]))
        byte_array.append(','.join(str(b) for b in w.tobytes()))

    dtype_names=[f'const char *{i}_dtype="{j}"; ' for i,j in zip(name_ls,dtype_ls)]
    dtype_names_str = '\n'.join([x for x in dtype_names])

    dim_names=[f'const char {i}_dim={j};' for i,j in zip(name_ls,dim_ls)]
    dim_names_str = '\n'.join([x for x in dim_names])

    shape_names=[f'const uint32_t {i}_shape[] = {{{j}}};' for i,j in zip(name_ls, shape_ls)]
    shape_names_str = '\n'.join([x for x in shape_names])

    #data_names=[f'const unsigned char {i}_data[]={{{j}}}; ' for i,j in zip(name_ls,byte_array)]
    data_names=[f'const uint8_t {i}_data[]={{{j}}}; ' for i,j in zip(name_ls,byte_array)]
    data_names_str = '\n'.join([x for x in data_names])

    size_names=[f'const uint64_t {i}_size = sizeof({i}_data);' for i,j in zip(name_ls, byte_array)]
    size_names_str = '\n'.join([x for x in size_names])

    print(dtype_names_str)
    print(dim_names_str)

    print(shape_names_str)
    #print(data_names_str)
    print(size_names_str)

    cpp_data=f"""
#include "{filename + '.h'}"
{dtype_names_str}
{dim_names_str}
{shape_names_str}
{data_names_str}
{size_names_str}
    """
    dtype_names=[f'extern const char *{i}_dtype; ' for i,j in zip(name_ls,dtype_ls)]
    dtype_names_str = '\n'.join([x for x in dtype_names])


    dim_names=[f'extern const char {i}_dim;' for i,j in zip(name_ls,dim_ls)]
    dim_names_str = '\n'.join([x for x in dim_names])

    
    shape_names=[f'extern const uint32_t {i}_shape[] ;' for i,j in zip(name_ls, shape_ls)]
    shape_names_str = '\n'.join([x for x in shape_names])

    size_names=[f'extern const uint64_t {i}_size ;' for i,j in zip(name_ls, data_names)]
    size_names_str = '\n'.join([x for x in size_names])

    
    #data_names=[f'extern const unsigned char {i}_data[]; ' for i,j in zip(name_ls,data_names)]
    data_names=[f'extern const uint8_t {i}_data[]; ' for i,j in zip(name_ls,data_names)]
    data_names_str = '\n'.join([x for x in data_names])


    with open(filename + '.c', 'w') as cpp_file:
        cpp_file.write(cpp_data)

    header_data=f"""

#ifndef {filename.upper()}_H_
#define {filename.upper()}_H_

#include <stdint.h>
#include "tensor1.h"
{dtype_names_str}
{dim_names_str}
{shape_names_str}
{size_names_str}
{data_names_str}

//extern const unsigned char model_array[];
//extern const int model_array_len;

#endif  // {filename.upper()}_H_
    """

    with open( filename + '.h', 'w') as header_file:
        header_file.write(header_data)

    print(f"Length of CPP byte array: {len(byte_array)}")
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--file_in', type=str, default=None)
    parser.add_argument('--file_out', type=str, default=None)
    parser.add_argument('--hidden_size', type=int, default=None)
    args = parser.parse_args()
    if  args.file_out is None: 
        print("Set Arguments!")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = NetSparse(args).to(device)
    #model.load_state_dict(torch.load(f'../weights/{args.file_in}.pt'))
    
    state_dict=torch.load(f'../weights/mnist_tiled_nn_128_packed.pt')

    model_np={}
    for name, param_tensor in torch.load('../weights/mnist_tiled_nn_128_packed.pt').items():
        print(name, "\t", param_tensor.size(), "\t",param_tensor.cpu().detach().numpy().dtype)
        model_np[name]=param_tensor.cpu().detach().numpy()
    write_c_blob(model_np, f'{args.file_out}_weights')
    #write_blob(model_np, f'{args.file_out}_{args.hidden_size}')

if __name__ == '__main__':
    main()