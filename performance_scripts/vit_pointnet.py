import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
import seaborn as sns

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

sns.set(style="whitegrid")
# Create the first figure (left subplot)
plt.figure(figsize=(12, 6), )  # Adjust the figure size as needed
plt.subplot(1, 2, 1)




#plt.grid(True, linestyle='--', alpha=0.9)


# to change default colormap
plt.rcParams["image.cmap"] = "Dark2"

colors =['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E']

json_file_path = base_dir+'performance_scripts/perf_log/kernel_standard/model_tiled_vit/log_True_compress4/maserati_1971668.1705079008959759065.pt.trace.json'
memory_data1 = extract_memory_data(json_file_path)

json_file_path = base_dir+'performance_scripts/perf_log/kernel_tiled/model_tiled_vit/log_True_compress4/maserati_1972253.1705079275990852903.pt.trace.json'
memory_data2= extract_memory_data(json_file_path)

json_file_path = base_dir+'performance_scripts/perf_log/kernel_tiled/model_tiled_vit/log_True_compress8/maserati_1972305.1705079305498096474.pt.trace.json'
memory_data3= extract_memory_data(json_file_path)


print(f'standard: {max(memory_data1)}')
print(f'tiled 4: {max(memory_data2)}')


max_len = max(len(memory_data1), len(memory_data2), len(memory_data3))
memory_data1 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(memory_data1)), memory_data1)
memory_data2 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(memory_data2)), memory_data2)
memory_data3 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(memory_data3)), memory_data3)

print(f'standard: {max(memory_data1)}')
print(f'tiled 4: {max(memory_data2)}')




max_len = max(len(memory_data1), len(memory_data2), len(memory_data3))
memory_data1 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(memory_data1)), memory_data1)
memory_data2 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(memory_data2)), memory_data2)
memory_data3 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(memory_data3)), memory_data3)


x_values = list(range(1, len(memory_data1) + 1))
plt.plot(x_values, memory_data1, marker='.', linestyle='-', color=colors[0], label='Full Precision')

x_values = list(range(1, len(memory_data2) + 1))
plt.plot(x_values, memory_data2, marker='.', linestyle='-', color=colors[1], label='F.P. + Tiled (p=4)')

x_values = list(range(1, len(memory_data3) + 1))
plt.plot(x_values, memory_data3, marker='.', linestyle='-', color=colors[2], label='F.P. + Tiled (p=8)')

plt.xlim(0,x_values[-1])

#plt.patch.set_edgecolor('gray')

plt.xlabel('Time Step', fontsize=16)
plt.ylabel('Memory (MB)' ,fontsize=16)
plt.yticks(fontsize=12)
plt.title('ViT, CIFAR-10', fontsize=16)
#plt.legend(fontsize=18)



#sns.set(style="whitegrid")

#plt.gcf().set_edgecolor('gray')
















# Create the second figure (right subplot)
plt.subplot(1, 2, 2)
#plt.edgecolor('gray')

#plt.grid(True, linestyle='--', alpha=0.9)



json_file_path = base_dir+'performance_scripts/perf_log/kernel_standard/model_tiled_pointnet/log_True_compress4/maserati_2053964.1705130654239021888.pt.trace.json'
memory_data1 = extract_memory_data(json_file_path)

json_file_path = base_dir+'performance_scripts/perf_log/kernel_tiled/model_tiled_pointnet/log_True_compress4/maserati_2053905.1705130623169242074.pt.trace.json'
memory_data2= extract_memory_data(json_file_path)

json_file_path = base_dir+'performance_scripts/perf_log/kernel_tiled/model_tiled_pointnet/log_True_compress8/maserati_2053858.1705130608893849473.pt.trace.json'
memory_data3= extract_memory_data(json_file_path)

max_len = max(len(memory_data1), len(memory_data2), len(memory_data3))
memory_data1 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(memory_data1)), memory_data1)
memory_data2 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(memory_data2)), memory_data2)
memory_data3 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(memory_data3)), memory_data3)

print(f'standard: {max(memory_data1)}')
print(f'tiled 4: {max(memory_data2)}')

x_values = list(range(1, len(memory_data1) + 1))
plt.plot(x_values, memory_data1, marker='.', linestyle='-', color=colors[0], label='Full Precision')

x_values = list(range(1, len(memory_data2) + 1))
plt.plot(x_values, memory_data2, marker='.', linestyle='-', color=colors[1], label='F.P. + Tiled (p=4)')

x_values = list(range(1, len(memory_data3) + 1))
plt.plot(x_values, memory_data3, marker='.', linestyle='-', color=colors[2], label='F.P. + Tiled (p=8)')

plt.xlim(0,x_values[-1])
plt.yticks(fontsize=12)
plt.xlabel('Time Step', fontsize=16)

plt.title('PointNet', fontsize=16)


#sns.set(style="whitegrid")
#plt.gcf().set_edgecolor('gray')

plt.tight_layout()

#plt.subplots_adjust(wspace=0.1)

plt.legend(loc='lower center', bbox_to_anchor=(-0.1, -.255), ncol=3, fontsize=16)
plt.subplots_adjust(bottom=0.18)
plt.savefig('vit_pointnet.pdf')
