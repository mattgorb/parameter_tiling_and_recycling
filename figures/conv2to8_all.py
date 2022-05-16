import matplotlib.pyplot as plt
import seaborn as sns


col=4
plt.clf()
sns.set_style('whitegrid')
fig, axs = plt.subplots(1, col, sharex=True, figsize=(5*col,6.25))

sns.lineplot(x=[50,90,80,60,40,20], y=[79.56,70.1,77.9,79.98,78.3,64.1] ,ax=axs[0], linestyle='-.',label='Biprop', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[79.69,64.05,75.15,79.1,78.6,70.06] ,ax=axs[0],linestyle='-.', label='Edge-Popup', legend=False)
#axs[0].errorbar([50,90,80,60,40,20], [81.68,70.22,79.06,81.12,81.8,81.05] , label='Edge-Popup+Recycle (Ours)',linewidth =2,  )
#axs[0].errorbar([50,90,80,60,40,20], [81.36,76.9,80.23,81.36,81.2,80.28] , label='Biprop+Recycle (Ours)',linewidth =2,  )
axs[0].errorbar([20,40,50,60,80,90], [81.05,81.6,81.68,81.12,79.06,70.22] ,[0.2,0.22,.9,.3,.65,.66], label='Edge-Popup+Recycle',linewidth =2,  )
axs[0].errorbar([20,40,50,60,80,90], [80.28,81.2,81.36,81.36,80.23, 76.9, ] ,[0.33,.67,0.77,0.344,0.54,0.52], label='Biprop+Recycle',linewidth =2,  )
axs[0].axhline(79.9, linestyle=':',label='Baseline (Learned Weights)' , linewidth=2.5, color='c', )
axs[0].set_title(label='Conv-2', fontdict = {'fontsize' : 22})

sns.lineplot(x=[50,90,80,60,40,20], y=[87.42,79,85.34,87.2,85.9,74.71] ,ax=axs[1],linestyle='-.', label='Biprop', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[86.67,74.42,83.28,86.43,86.74,77.6] ,ax=axs[1],linestyle='-.', label='Edge-Popup', legend=False)
#axs[1].errorbar([50,90,80,60,40,20], [88.5,79.78,86.9,88.15,89.2,88.95] ,[0,0,0,0,0,0], label='Edge-Popup+Recycle (Ours)', )
#axs[1].errorbar([50,90,80,60,40,20], [88.91,84.3,87.75,88.64,88.8,88.05] ,[0,0,0,0,0,0], label='Biprop+Recycle (Ours)', )
axs[1].errorbar([20,40,50,60,80,90], [88.95,88.9, 88.5,88.15, 86.9, 79.78] ,[0.26,0.37,0.44,0.44,0.5,0.7], label='Edge-Popup+Recycle', )
axs[1].errorbar([20,40,50,60,80,90], [88.05,88.8,88.91,88.64, 87.75, 84.3] ,[0.2,0.2,0.7,0.3,0.7,0.7], label='Biprop+Recycle', )
axs[1].axhline(87, linestyle=':',linewidth=2.5, color='c', )
axs[1].set_title(label='Conv-4', fontdict = {'fontsize' : 22})


sns.lineplot(x=[50,90,80,60,40,20], y=[89.52,82.38,87.5,90.25,88.9,78.2] ,ax=axs[2],linestyle='-.', label='Biprop', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[89.53,79.57,85.4,89.1,89.33,83] ,ax=axs[2],linestyle='-.', label='Edge-Popup', legend=False)
#axs[2].errorbar([50,90,80,60,40,20], [90.9,83.2,88.6,90.4,91.2,91.1] ,[2.2,2.1,0,0,0,0], label='Edge-Popup+Recycle (Ours)', )
#axs[2].errorbar([50,90,80,60,40,20], [90.9,85.5,90,90.6,90.8,90.2] , [2.3,2.13,0,0,0,0],label='Biprop+Recycle (Ours)', )
axs[2].errorbar([20,40,50,60,80,90], [91.1,91.2,90.9, 90.4, 88.6, 83.2] ,[.87,.76,.5,0.32,0.33,0.5], label='Edge-Popup+Recycle', )
axs[2].errorbar([20,40,50,60,80,90], [90.2,90.8,90.9,90.6,90,85.5] , [.88,.74,.55,0.37,0.3,0.35],label='Biprop+Recycle', )
axs[2].axhline(89, linestyle=':', linewidth=2.5, color='c', )
axs[2].set_title(label='Conv-6', fontdict = {'fontsize' : 22})

sns.lineplot(x=[50,90,80,60,40,20], y=[90.35,84.84,89.11,90.54,89.13,76.59] ,ax=axs[3], label='Biprop',linestyle='-.', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[90.28,83.6,86.9,90.5,90.15,84.39] ,ax=axs[3],linestyle='-.', label='Edge-Popup', legend=False)
#axs[3].errorbar([50,90,80,60,40,20], [91.36,86.93,89.87,90.95,91.5,91.15] ,[0,0,0,0,0,0], label='Edge-Popup+Recycle (Ours)', )
#axs[3].errorbar([50,90,80,60,40,20], [91,86.4,90.55,90.85,90.93,90.35] ,[0,0,0,0,0,0], label='Biprop+Recycle (Ours)',)
axs[3].errorbar([20,40,50,60,80,90], [91.15,91.5,91.36, 90.95,89.87, 86.93] ,[0.3,0.33,0.36,0.34,.4,0.22], label='Edge-Popup+Recycle', )
axs[3].errorbar([20,40,50,60,80,90], [90.15,91.,91.6, 91.2, 90.55, 86.4] ,[0.2,0.34,0.35,0.2,0.53,0.27], label='Biprop+Recycle',)
axs[3].axhline(89.41, linestyle=':' ,linewidth=2.5, color='c', )
axs[3].set_title(label='Conv-8', fontdict = {'fontsize' : 22})


for ax in axs:
    ax.set_xlabel("Percentage of Weights Pruned", fontdict = {'fontsize' : 17})
    #ax.get_legend().remove()
axs[0].set_ylabel("CIFAR-10 Test Accuracy", fontdict = {'fontsize' : 24})

#plt.xticks([.2,.5, .5, .6,.8,.9])
plt.xticks([20, 40,50,60,80, 90])
plt.xlim([19.9, 90.1])
handles, labels = axs[0].get_legend_handles_labels()
print(labels)
handles=[handles[0], handles[4], handles[1], handles[3], handles[2]]
labels=[labels[0],labels[4], labels[1], labels[3],  labels[2]]
fig.legend(handles, labels, loc='lower left', ncol=5,bbox_to_anchor=(.04, .001), prop={'size': 22})



plt.tight_layout(rect=[0,.1,1,1])
#print(labels)

plt.savefig('figs/depth_all.pdf')






