import matplotlib.pyplot as plt
import seaborn as sns


col=4
plt.clf()
sns.set_style('whitegrid')
fig, axs = plt.subplots(1, col, sharex=True, figsize=(5*col,6.25))
sns.lineplot(x=[0.1,0.25,0.5,1], y=[47.2,64.55,74.69,79.56] ,ax=axs[0], linestyle='-.',label='Biprop', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[52.55,68.8,76.66,81.36] ,ax=axs[0], label='Biprop+Recycle', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[58.5,72.49,77.55,79.9] ,ax=axs[0],linestyle='--',label='Baseline (Learned Weights)', legend=False)
axs[0].set_title(label='Wide Conv-2', fontdict = {'fontsize' : 16})

sns.lineplot(x=[0.1,0.25,0.5,1], y=[50.4,73.23,82.6,87.42] ,ax=axs[1],linestyle='-.', label='Biprop', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[59.64,78.51,86.5,88.91] ,ax=axs[1], label='Biprop+Recycle', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[67.26,81.13,84.96,86.66] ,linestyle='--',ax=axs[1], label='Baseline (Learned Weights)', legend=False)
axs[1].set_title(label='Wide Conv-4', fontdict = {'fontsize' : 16})




sns.lineplot(x=[0.1,0.25,0.5,1], y=[56.9,77.84,86.23,89.52] ,ax=axs[2],linestyle='-.', label='Biprop', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[65,77.9,88.43,90.9] ,ax=axs[2], label='Biprop+Recycle', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[72.48,83.65,87.54,89.14] ,linestyle='--',ax=axs[2], label='Baseline (Learned Weights)', legend=False)
axs[2].set_title(label='Wide Conv-6', fontdict = {'fontsize' : 16})

sns.lineplot(x=[0.1,0.25,0.5,1], y=[61.2,80.4,87.33,90.35] ,ax=axs[3],linestyle='-.', label='Biprop', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[69,84.51,89.67,91] ,ax=axs[3], label='Biprop+Recycle', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[71.56,85.5,87.99,91.11] ,linestyle='--',ax=axs[3], label='Baseline (Learned Weights)', legend=False,)
#axs[3].title.set_text('Wide Conv-8',)
axs[3].set_title(label='Wide Conv-8', fontdict = {'fontsize' : 16})
#axs[3].axhline(89.41, linestyle=':', )

for ax in axs:
    ax.set_xlabel("Layer Width Factor", fontdict = {'fontsize' : 16})
    #ax.get_legend().remove()
axs[0].set_ylabel("CIFAR-10 Test Accuracy",fontdict = {'fontsize' : 16} )

plt.xticks([.1,.25, .5, 1])
plt.xlim([.1, 1])
#handles, labels = axs[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.12, .02))
handles, labels = axs[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.3, .01), prop={'size': 18})
plt.tight_layout(rect=[0,.1,1,1])

#plt.show()
plt.savefig('width_biprop.pdf')

del fig
del axs





