import matplotlib.pyplot as plt
import seaborn as sns


col=4
plt.clf()
sns.set_style('whitegrid')
fig, axs = plt.subplots(1, col, sharex=True, figsize=(5*col,6.25))

sns.lineplot(x=[50,90,80,60,40,20], y=[79.56,70.1,77.9,79.98,78.3,64.1] ,ax=axs[0], linestyle='-.',label='Biprop', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[79.69,64.05,75.15,79.1,78.6,70.06] ,ax=axs[0],linestyle='-.', label='Edge-Popup', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[81.36,76.9,80.23,81.36,81.2,80.28] ,ax=axs[0], label='Biprop+Recycle (Ours)', legend=False)
axs[0].axhline(79.9, linestyle=':',label='Baseline (Learned Weights)' , linewidth=2., color='deepskyblue')
axs[0].set_title(label='Conv-2', fontdict = {'fontsize' : 18})

sns.lineplot(x=[50,90,80,60,40,20], y=[87.42,79,85.34,87.2,85.9,74.71] ,ax=axs[1],linestyle='-.', label='Biprop', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[86.67,74.42,83.28,86.43,86.74,77.6] ,ax=axs[1],linestyle='-.', label='Edge-Popup', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[88.91,84.3,87.75,88.64,88.8,88.05] ,ax=axs[1], label='Biprop+Recycle (Ours)', legend=False)

axs[1].axhline(87, linestyle=':',linewidth=2., color='deepskyblue' )
axs[1].set_title(label='Conv-4', fontdict = {'fontsize' : 18})

sns.lineplot(x=[50,90,80,60,40,20], y=[89.52,82.38,87.5,90.25,88.9,78.2] ,ax=axs[2],linestyle='-.', label='Biprop', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[89.53,79.57,85.4,89.1,89.33,83] ,ax=axs[2],linestyle='-.', label='Edge-Popup', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[90.9,85.5,90,90.6,90.8,90.2] ,ax=axs[2], label='Biprop+Recycle (Ours)', legend=False)
axs[2].axhline(89, linestyle=':', linewidth=2., color='deepskyblue')
axs[2].set_title(label='Conv-6', fontdict = {'fontsize' : 18})

sns.lineplot(x=[50,90,80,60,40,20], y=[90.35,84.84,89.11,90.54,89.13,76.59] ,ax=axs[3], label='Biprop',linestyle='-.', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[90.28,83.6,86.9,90.5,90.15,84.39] ,ax=axs[3],linestyle='-.', label='Edge-Popup', legend=False)
sns.lineplot(x=[50,90,80,60,40,20], y=[91,86.4,90.55,90.85,90.93,90.35] ,ax=axs[3], label='Biprop+Recycle (Ours)', legend=False)
axs[3].axhline(89.41, linestyle=':' ,linewidth=2., color='deepskyblue')
axs[3].set_title(label='Conv-8', fontdict = {'fontsize' : 18})


for ax in axs:
    ax.set_xlabel("Percentage of Weights Pruned", fontdict = {'fontsize' : 18})
    #ax.get_legend().remove()
axs[0].set_ylabel("CIFAR-10 Test Accuracy", fontdict = {'fontsize' : 18})

#plt.xticks([.2,.5, .5, .6,.8,.9])
plt.xticks([20, 40,50,60,80, 90])
plt.xlim([20, 90])
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', ncol=4,bbox_to_anchor=(.20, .0025), prop={'size': 18})
#handles, labels = axs[2].get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.62, .02))
plt.tight_layout(rect=[0,.08,1,1])
#print(labels)

plt.savefig('figs/conv2to8_biprop_3.pdf')

del fig
del axs





