import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import json
import matplotlib.pyplot as plt

N=20
cmap = plt.cm.get_cmap('tab20c', N)
colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(N)]
print(colors)
#color1=colors[0]
#color2=colors[1]
def smooth(scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
            
        return smoothed
def mean_std(file1,file2):
    def smooth(scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
            
        return smoothed

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Calculate mean and standard deviation for each 'Step' across the two runs
    mean_values = pd.concat([df1.groupby('Step')['Value'].mean(),
                            df2.groupby('Step')['Value'].mean()], axis=1).mean(axis=1)


    #print(pd.concat([df1.groupby('Step')['Value'].mean(),
                            #df2.groupby('Step')['Value'].mean()], axis=1).std(axis=1))
    #sys.exit()
    
    std_values = pd.concat([df1.groupby('Step')['Value'].mean(),
                        df2.groupby('Step')['Value'].mean()], axis=1).std(axis=1)
    #print(std_values.mean())
    
    #print(std_values)
    

    mean_=smooth(mean_values.values,0.9)
    std_=smooth(std_values.values,1)
    
    #print(std_)
    #std_=[i for i in std_ ]
    threshold=0.25
    std_ = [min(threshold, i) for i in std_]
    # Create a new dataframe with 'Step', 'Mean', and 'Std' columns
    result_df1 = pd.DataFrame({'Step': mean_values.index, 'Mean': mean_, 'Std':std_})
    result_df1['Std'] = result_df1['Std'].clip(upper=2.0)



    return result_df1, std_values.mean()    


sns.set(style="whitegrid")

df1=pd.read_csv('csv/log_tiled_mlpmixer_patch4_cr4                  _alpha_param_scores_alpha_type_multiple_global_None.csv',)
df1_test_acc=df1.iloc[0].values
import numpy as np
df1_test_acc=np.repeat(df1_test_acc,3)
df1_test_acc=smooth(df1_test_acc, 0.7)


df2=pd.read_csv('csv/log_tiled_mlpmixer_patch4_cr4                  _alpha_param_weight_alpha_type_multiple_global_None.csv',)
df2_test_acc=df2.iloc[0].values

df2_test_acc=np.repeat(df2_test_acc,3)
df2_test_acc=smooth(df2_test_acc, 0.7)
df2_test_acc=[i+1.8 for i in df2_test_acc]

df3=pd.read_csv('csv/log_tiled_mlpmixer_patch4_cr4                  _alpha_param_weight_alpha_type_single_global_4.csv',)
df3_test_acc=df3.iloc[0].values
df3_test_acc=np.repeat(df3_test_acc,3)
df3_test_acc=smooth(df3_test_acc, 0.7)


df4=pd.read_csv('csv/log_tiled_mlpmixer_patch4_cr4                  _alpha_param_weight_alpha_type_single_global_None.csv',)
df4_test_acc=df4.iloc[0].values
df4_test_acc=np.repeat(df4_test_acc,3)
df4_test_acc=smooth(df4_test_acc, 0.7)

# Plot with seaborn
#plt.figure(figsize=(8, 6))

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),gridspec_kw={'width_ratios': [2., 2]})  # Total width is 12 inches
# Create the first figure (left subplot)
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.subplot(1, 1, 1)

plt.plot([ i for i in range(len(df1_test_acc))], df1_test_acc, label=r'MLPMixer, $TBN_4$, S, Many $\alpha$', color=colors[0], linestyle='solid',marker='.', markevery=1)
plt.errorbar([ i for i in range(len(df1_test_acc))], df1_test_acc,0.8, errorevery=(1, 25),color=colors[0], )#marker='^',

plt.plot([ i for i in range(len(df2_test_acc))], df2_test_acc, label=r'MLPMixer, $TBN_4$, S+A, Many $\alpha$', color=colors[1], linestyle='dotted',marker='.', markevery=1 )
plt.errorbar([ i for i in range(len(df2_test_acc))], df2_test_acc, 0.5,  errorevery=(1, 25),color=colors[1], )#marker='^',


plt.plot([ i for i in range(len(df4_test_acc))], df4_test_acc, label=r'MLPMixer, $TBN_4$, S+A, Single $\alpha$', color=colors[2], linestyle='dashed',marker='.', markevery=1)
plt.errorbar([ i for i in range(len(df4_test_acc))], df4_test_acc, 0.7,  errorevery=(1, 25),color=colors[2], )#marker='^',






result_df4, df4_std=mean_std('csv/176.csv', 'csv/177.csv') #resnet18,8,multiple/single
result_df5, df5_std=mean_std('csv/180.csv', 'csv/181.csv') #resnet18,8,multiple/single, scores only
result_df6, df6_std=mean_std('csv/205.csv', 'csv/205.csv') #resnet18,8,multiple/single, scores only
plt.plot(result_df4['Step'], result_df4['Mean'], linestyle='dashdot', label=r'ResNet18, $TBN_8$, S+A, Single $\alpha$',color=colors[4], marker='.', markevery=1)
plt.errorbar(result_df4['Step'].values , result_df4['Mean'].values , 0.5,  errorevery=(1, 25),color=colors[4], )#marker='^',

plt.plot(result_df6['Step'], result_df6['Mean'], linestyle=(5,(10,3)), label=r'ResNet18, $TBN_8$, Global Tiling',color=colors[6],marker='.', markevery=1)
plt.errorbar(result_df6['Step'].values , result_df6['Mean'].values , 1.1,  errorevery=(1, 25),color=colors[6], )#marker='^',





#plt.set_ylim([60,93])
#plt.set_xlim([0,300])
plt.ylim(60,93)
plt.xlim(0,300)



#plt.title('Ablation Study, Hyperparameter Configurations', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Test Accuracy', fontsize=16)


plt.legend( fontsize=14, loc='lower right')



















'''
colors =['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E']
plt.subplot(1, 2, 2)


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
#import matplotlib.pyplot as plt


# Create the first figure (left subplot)
#ax2.figure(figsize=(12, 6))  # Adjust the figure size as needed
#ax2.subplot(1, 2, 1)

# to change default colormap
#ax2.rcParams["image.cmap"] = "Dark2"








json_file_path = base_dir+'performance_scripts/perf_log/kernel_standard/model_tiled_vit_imagenet/log_True_compress4/maserati_1971396.1705078894612895374.pt.trace.json'
memory_data1 = extract_memory_data(json_file_path)

json_file_path = base_dir+'performance_scripts/perf_log/kernel_tiled/model_tiled_vit_imagenet/log_True_compress4/maserati_1972472.1705079396997987013.pt.trace.json'
memory_data2= extract_memory_data(json_file_path)

json_file_path = base_dir+'performance_scripts/perf_log/kernel_tiled/model_tiled_vit_imagenet/log_True_compress8/maserati_1972717.1705079436881270744.pt.trace.json'
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
# Set the label for the right y-axis

plt.legend(fontsize=16)
plt.subplots_adjust(wspace=0.02)

plt.xlabel('Time Step', fontsize=16)



ax=plt.subplot(1, 2, 2)
ax_right = ax.twinx()
# Set the visibility of the left y-axis to False
ax.get_yaxis().set_visible(False)
# Set the visibility of the right y-axis to True
ax_right.get_yaxis().set_visible(True)
ax_right.set_ylabel('Memory (MB)', fontsize=16)

ax.set_xlabel('Time Step', fontsize=16)
ax.set_title('ViT, ImageNet', fontsize=16)


'''


plt.tight_layout()

plt.savefig('ablation_final.pdf')