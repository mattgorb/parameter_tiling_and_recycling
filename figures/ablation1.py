import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib



N=20
cmap = plt.cm.get_cmap('tab20c', N)
colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(N)]
print(colors)
#color1=colors[0]
#color2=colors[1]

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
result_df1, df1_std=mean_std('csv/169.csv', 'csv/171.csv') #resnet18, 4, single
result_df2, df2_std=mean_std('csv/172.csv', 'csv/172.csv') #resnet18, 4, multiple & single, scores only
result_df3, df3_std=mean_std('csv/174.csv', 'csv/174.csv',) #resnet18, 4, global

#8
result_df4, df4_std=mean_std('csv/176.csv', 'csv/177.csv') #resnet18,8,multiple/single
result_df5, df5_std=mean_std('csv/180.csv', 'csv/181.csv') #resnet18,8,multiple/single, scores only
result_df6, df6_std=mean_std('csv/205.csv', 'csv/205.csv') #resnet18,8,multiple/single, scores only

print(df1_std)

# Plot with seaborn
plt.figure(figsize=(8, 6))

plt.plot(result_df1['Step'], result_df1['Mean'], label='1', color=colors[0])
#plt.fill_between(result_df1['Step'], result_df1['Mean'] - result_df1['Std'], result_df1['Mean'] + result_df1['Std'], alpha=0.5,color=colors[0] )
plt.errorbar(result_df1['Step'].values , result_df1['Mean'].values , 0.6, linestyle='None', errorevery=(1, 25))#marker='^',
#plt.axhline(y=result_df1['Mean'] + df1_std, color='r', linestyle='--', label='Mean + Std Dev')
#plt.axhline(y=result_df1['Mean'] - df1_std, color='r', linestyle='--', label='Mean - Std Dev')



plt.plot(result_df2['Step'], result_df2['Mean'], label='2',color=colors[1])
#plt.errorbar(result_df2['Step'].values , result_df2['Mean'].values , 0.15, linestyle='None', errorevery=(1, 150),color=colors[1])#marker='^',
#plt.fill_between(result_df2['Step'], result_df2['Mean'] - result_df2['Std'], result_df2['Mean'] + result_df2['Std'], alpha=0.5,color=colors[1])

plt.plot(result_df3['Step'], result_df3['Mean'], label='2',color=colors[2])
#plt.errorbar(result_df3['Step'].values , result_df3['Mean'].values , 0.7, linestyle='None', errorevery=(1, 150),color=colors[2])#marker='^',
#plt.fill_between(result_df3['Step'], result_df3['Mean'] - result_df3['Std'], result_df3['Mean'] + result_df3['Std'], alpha=0.5,color=colors[2])






plt.plot(result_df4['Step'], result_df4['Mean'], label='3',color=colors[4])
plt.errorbar(result_df4['Step'].values , result_df4['Mean'].values , 0.5, linestyle='None', errorevery=(1, 25),color=colors[4])#marker='^',
#plt.fill_between(result_df4['Step'], result_df4['Mean'] - result_df4['Std'], result_df4['Mean'] + result_df4['Std'], alpha=0.5,color=colors[4])

#plt.plot(result_df5['Step'], result_df5['Mean'], label='4',color=colors[5])
#plt.fill_between(result_df5['Step'], result_df5['Mean'] - result_df5['Std'], result_df5['Mean'] + result_df5['Std'], alpha=0.5,color=colors[5])

plt.plot(result_df6['Step'], result_df6['Mean'], label='5',color=colors[6])
#plt.fill_between(result_df6['Step'], result_df6['Mean'] - result_df6['Std'], result_df6['Mean'] + result_df6['Std'], alpha=0.5,color=colors[6])


plt.ylim(70,94)
plt.xlim(25,300)



plt.title('Mean Test Accuracy per Epoch with Standard Deviation')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')





# Manually set legend handles and labels
#handles, labels = plt.gca().get_legend_handles_labels()
#handles = [handles[0], handles[1]]
#labels = ['1'] * 2
# Add legend with modified handles and labels
#plt.legend(handles, labels)

plt.legend()
sns.set(style="whitegrid")
plt.grid(True)
plt.savefig('ablation1.png')