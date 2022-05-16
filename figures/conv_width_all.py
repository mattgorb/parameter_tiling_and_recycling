import matplotlib.pyplot as plt
import seaborn as sns


col=4
plt.clf()
sns.set_style('whitegrid')
fig, axs = plt.subplots(1, col, sharex=True, figsize=(5*col,6.25))
sns.lineplot(x=[0.1,0.25,0.5,1], y=[47.2,64.55,74.69,79.56] ,ax=axs[0], linestyle='-.',label='Biprop', legend=False)
axs[0].errorbar([0.1,0.25,0.5,1], [52.55,68.8,76.66,81.36] ,[.8,.67,.15,.2], label='Biprop+Recycle',linewidth =2, )
sns.lineplot(x=[0.1,0.25,0.5,1], y=[48.1,64.52,73.65,79.69] ,ax=axs[0], linestyle='-.',label='Edge-Popup', legend=False)
axs[0].errorbar([0.1,0.25,0.5,1], [53.4,70.33,77,81.68] ,[.76,.56,.2,.3], label='Edge-Popup+Recycle', )
sns.lineplot(x=[0.1,0.25,0.5,1], y=[58.5,72.49,77.55,79.9] ,ax=axs[0],linestyle='--',label='Baseline (Learned Weights)', legend=False)
axs[0].set_title(label='Wide Conv-2', fontdict = {'fontsize' : 22})


sns.lineplot(x=[0.1,0.25,0.5,1], y=[50.4,73.23,82.6,87.42] ,ax=axs[1],linestyle='-.', label='Biprop', legend=False)
axs[1].errorbar([0.1,0.25,0.5,1], [59.64,78.51,86.5,88.91] , [2,1,.3,.67], label='Biprop+Recycle',linewidth =2, )
sns.lineplot(x=[0.1,0.25,0.5,1], y=[50.73,73.34,82.08,86.67] , ax=axs[1],linestyle='-.', label='Edge-Popup', legend=False)
axs[1].errorbar([0.1,0.25,0.5,1], [59.82,78.92,85.63,88.5] ,[1.5,1,.25,.5], label='Edge-Popup+Recycle',linewidth =2, )
sns.lineplot(x=[0.1,0.25,0.5,1], y=[67.26,81.13,84.96,86.66] ,linestyle='--',ax=axs[1], label='Baseline (Learned Weights)', legend=False)

axs[1].set_title(label='Wide Conv-4', fontdict = {'fontsize' : 22})




sns.lineplot(x=[0.1,0.25,0.5,1], y=[56.9,77.84,86.23,89.52] ,ax=axs[2],linestyle='-.', label='Biprop', legend=False)
axs[2].errorbar([0.1,0.25,0.5,1], [65,80.9,88.43,90.9] ,[.2,.6,.4,.52], label='Biprop+Recycle',linewidth =2, )
sns.lineplot(x=[0.1,0.25,0.5,1], y=[57.3,78.48,86.24,89.53] ,ax=axs[2],linestyle='-.', label='Edge-Popup', legend=False)
axs[2].errorbar([0.1,0.25,0.5,1], [64.1,83,88.57,90.9] ,[2,.4,.22,.47], label='Edge-Popup+Recycle',linewidth =2, )
sns.lineplot(x=[0.1,0.25,0.5,1], y=[72.48,83.65,87.54,89.14] ,linestyle='--',ax=axs[2], label='Baseline (Learned Weights)', legend=False)
axs[2].set_title(label='Wide Conv-6', fontdict = {'fontsize' : 22})

sns.lineplot(x=[0.1,0.25,0.5,1], y=[61.2,80.4,87.33,90.35] ,ax=axs[3],linestyle='-.', label='Biprop', legend=False)
axs[3].errorbar([0.1,0.25,0.5,1], [69,84.51,89.67,91] ,[2.5,.58,.29,.12], label='Biprop+Recycle',linewidth =2, )
sns.lineplot(x=[0.1,0.25,0.5,1], y=[63.35,81.5,87.92,90.28] ,ax=axs[3],linestyle='-.', label='Edge-Popup', legend=False)
axs[3].errorbar([0.1,0.25,0.5,1], [70,85.5,89.73,91.36] ,[2.23,.67,.21,.14], label='Edge-Popup+Recycle', linewidth =2, )
sns.lineplot(x=[0.1,0.25,0.5,1], y=[71.56,85.5,87.99,91.11] ,linestyle='--',ax=axs[3], label='Baseline (Learned Weights)', legend=False,)
axs[3].set_title(label='Wide Conv-8', fontdict = {'fontsize' : 22})

for ax in axs:
    ax.set_xlabel("Layer Width Factor", fontdict = {'fontsize' : 17})
    #ax.get_legend().remove()
axs[0].set_ylabel("CIFAR-10 Test Accuracy",fontdict = {'fontsize' : 22} )

plt.xticks([.1,.25, .5, 1])
plt.xlim([.098, 1.002])
#handles, labels = axs[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.12, .02))
handles, labels = axs[1].get_legend_handles_labels()

handles=[handles[0], handles[3], handles[1], handles[4], handles[2]]
labels=[labels[0], labels[3], labels[1], labels[4], labels[2]]
print(handles)
print(labels)
fig.legend(handles, labels, loc='lower left', ncol=5,bbox_to_anchor=(.04, .001), prop={'size': 22})
plt.tight_layout(rect=[0,.1,1,1])

#plt.show()
plt.savefig('figs/width_biprop.pdf')

del fig
del axs





