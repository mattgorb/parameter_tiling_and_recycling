import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib



N=10
cmap = plt.cm.get_cmap('tab10', N)
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


'''
TODO:
add mlpmixer, resnet50 
add global compression 
figure 1: resnet50, mlpmixer (both have 600+epochs)
figure 2: both resnet18's

TODO this fig: 
add global compression for both resnet18 8 and 4.
change colors

resnet18-8-global 205.txt
resnet18-4-global 174,175.txt
'''
#result_df1, df1_std=mean_std('csv/169.csv', 'csv/171.csv') #resnet18, 4, single
#result_df2, df2_std=mean_std('csv/172.csv', 'csv/172.csv') #resnet18, 4, multiple & single, scores only
#result_df3, df3_std=mean_std('csv/174.csv', 'csv/174.csv',) #resnet18, 4, global

#8
#result_df4, df4_std=mean_std('csv/176.csv', 'csv/177.csv') #resnet18,8,multiple/single
#result_df5, df5_std=mean_std('csv/180.csv', 'csv/181.csv') #resnet18,8,multiple/single, scores only
#result_df6, df6_std=mean_std('csv/205.csv', 'csv/205.csv') #resnet18,8,multiple/single, scores only

df1=pd.read_csv('csv/resnet50_global.csv',)
df1_test_acc=df1['Value'].values

df2=pd.read_csv('csv/resnet50_scores_only_multiple.csv',)
df2_test_acc=df2['Value'].values
#df2_test_acc=[i+1 for i in df2_test_acc]

df3=pd.read_csv('csv/resnet50_scores_scores_multiple.csv',)
df3_test_acc=df3['Value'].values

df4=pd.read_csv('csv/resnet50_single_alpha.csv',)
df4_test_acc=df4['Value'].values

# Plot with seaborn
plt.figure(figsize=(10, 6))

plt.plot([ i for i in range(len(df1_test_acc))], smooth(df1_test_acc, 0.85), label='Global Tiling',color= colors[0], linestyle='dotted')
plt.errorbar([ i for i in range(len(df1_test_acc))], smooth(df1_test_acc, 0.95) , 0.01, linestyle='None', errorevery=(1, 100), color=colors[0])

plt.plot([ i for i in range(len(df2_test_acc))], smooth(df2_test_acc, 0.85), label='S Only, Multiple Alphas',color=colors[1], linestyle='dashed')
plt.errorbar([ i for i in range(len(df2_test_acc))], smooth(df2_test_acc, 0.85) , 0.008, linestyle='None', errorevery=(1, 100), color=colors[1])

plt.plot([ i for i in range(len(df3_test_acc))], smooth(df3_test_acc, 0.85), label='W+A'', Multiple Alpha ',color=colors[2] , linestyle='solid')
plt.errorbar([ i for i in range(len(df3_test_acc))], smooth(df3_test_acc, 0.85) , 0.0075, linestyle='None', errorevery=(1, 100), color=colors[2])

plt.plot([ i for i in range(len(df4_test_acc))], smooth(df4_test_acc, 0.85), label='W+A'', Single Alpha', color=colors[3], linestyle='dashdot')
plt.errorbar([ i for i in range(len(df4_test_acc))], smooth(df4_test_acc, 0.85) , 0.007, linestyle='None', errorevery=(1, 100), color=colors[3])


plt.ylim(0.225,0.6)
plt.xlim(0,800)



plt.title('ResNet50 on Different Tiling Configurations', fontsize=18)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Test Loss', fontsize=20)





# Manually set legend handles and labels
#handles, labels = plt.gca().get_legend_handles_labels()
#handles = [handles[0], handles[1]]
#labels = ['1'] * 2
# Add legend with modified handles and labels
#plt.legend(handles, labels)

plt.legend(fontsize=16)
sns.set(style="whitegrid")
plt.grid(True)
plt.tight_layout()

plt.savefig('ablation3.png')