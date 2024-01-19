import json
import matplotlib.pyplot as plt

def extract_memory_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    #index_to_extract =3 # Change this to the desired index

    values=[]
    # Iteratively extract the ith value from each memory entry and convert to MB
    data=data['traceEvents']

    
    for entry in data:
        if 'args' in entry:
            if "Total Allocated" in entry['args'] :
                if entry['args'][ "Device Id"]!=-1:
                    values.append(int(entry['args']["Total Allocated"])/(1024*1024))
    return values
# Example usage


base_dir=f'/s/chopin/l/grad/mgorb/parameter_tiling_and_recycling/'


#plt.figure(figsize=(6, 6))  #
import matplotlib.pyplot as plt


# Create the first figure (left subplot)
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
plt.subplot(1, 2, 1)

# to change default colormap
plt.rcParams["image.cmap"] = "Dark2"


json_file_path = base_dir+'performance_scripts/perf_log/kernel_standard/model_tiled_vit_imagenet/log_True_compress4/maserati_1971396.1705078894612895374.pt.trace.json'
memory_data = extract_memory_data(json_file_path)
x_values = list(range(1, len(memory_data) + 1))
plt.plot(x_values, memory_data, marker='.', linestyle='-')



json_file_path = base_dir+'performance_scripts/perf_log/kernel_tiled/model_tiled_vit_imagenet/log_True_compress4/maserati_1972472.1705079396997987013.pt.trace.json'
memory_data= extract_memory_data(json_file_path)
x_values = list(range(1, len(memory_data) + 1))
plt.plot(x_values, memory_data,linestyle='-')
plt.plot(x_values, memory_data, marker='.', linestyle='-')


json_file_path = base_dir+'performance_scripts/perf_log/kernel_tiled/model_tiled_vit_imagenet/log_True_compress8/maserati_1972717.1705079436881270744.pt.trace.json'
memory_data= extract_memory_data(json_file_path)
x_values = list(range(1, len(memory_data) + 1))
plt.plot(x_values, memory_data,linestyle='-')
plt.plot(x_values, memory_data, marker='.', linestyle='-')






plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Plot of Your Values')



















# Create the second figure (right subplot)
plt.subplot(1, 2, 2)



json_file_path = base_dir+'performance_scripts/perf_log/kernel_standard/model_tiled_pointnet/log_True_compress4/maserati_2053964.1705130654239021888.pt.trace.json'
memory_data = extract_memory_data(json_file_path)
x_values = list(range(1, len(memory_data) + 1))
plt.plot(x_values, memory_data, marker='.', linestyle='-')


json_file_path = base_dir+'performance_scripts/perf_log/kernel_tiled/model_tiled_pointnet/log_True_compress4/maserati_2053905.1705130623169242074.pt.trace.json'
memory_data= extract_memory_data(json_file_path)
x_values = list(range(1, len(memory_data) + 1))
plt.plot(x_values, memory_data,linestyle='-')
plt.plot(x_values, memory_data, marker='.', linestyle='-')


json_file_path = base_dir+'performance_scripts/perf_log/kernel_tiled/model_tiled_pointnet/log_True_compress8/maserati_2053858.1705130608893849473.pt.trace.json'
memory_data= extract_memory_data(json_file_path)
x_values = list(range(1, len(memory_data) + 1))
plt.plot(x_values, memory_data,linestyle='-')
plt.plot(x_values, memory_data, marker='.', linestyle='-')





plt.tight_layout()
plt.savefig('vit_im_pointnet.png')
