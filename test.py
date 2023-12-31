import torch

# Assuming you have a smaller tensor with size (64, 784)
small_tensor = torch.randn(64, 784)#.sign()
print(small_tensor.stride())

# Define the tiling factor
tile_factor = 4

# Calculate the new size and stride for the larger tensor
new_size = (64 * tile_factor, 784)
new_stride = (784, small_tensor.stride(1))
#new_stride = (193, small_tensor.stride(1))

print(new_stride)
# Manually calculate the new storage offset
storage_offset = 0  # You may need to adjust this based on your requirements

# Create a larger tensor using torch.as_strided
#larger_tensor = torch.as_strided(small_tensor, size=new_size, stride=new_stride, storage_offset=0)
larger_tensor = torch.as_strided(small_tensor, size=new_size, stride=(784, 1), storage_offset=0)

# Print or use the larger tensor as needed
print(larger_tensor[0,:20])
print(larger_tensor[1,:20])
print(larger_tensor[64, :20])