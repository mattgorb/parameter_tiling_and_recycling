import matplotlib.pyplot as plt
import seaborn as sns


col=4
plt.clf()
fig, axs = plt.subplots(1, col, sharex=True, figsize=(5*col,5))

sns.lineplot(x=[.5,.95,.9,0.8,.6,.4,.2], y=[79.56,56.6,70.1,77.9,79.98,78.3,64.1] ,ax=axs[0], linestyle='-.',label='Biprop', legend=False)
#sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[81.83,78.04,81.75,81.3,80.7] ,ax=axs[0], label='EPU+IteRand', legend=False)
sns.lineplot(x=[.5,.95,.9,0.8,.6,.4,.2], y=[81.36,65.11,76.9,80.23,81.36,81.2,80.28] ,ax=axs[0], label='Biprop+Recycle', legend=False)
axs[0].axhline(79.9, linestyle=':', )
axs[0].set_title(label='Conv-2', fontdict = {'fontsize' : 14})

sns.lineplot(x=[.5,.95,.9,0.8,.6,.4,.2], y=[87.42,64.83,79,85.34,87.2,85.9,74.71] ,ax=axs[1],linestyle='-.', label='Biprop', legend=False)
#sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[91.11,88.42,91.16,91.55,90.99] ,ax=axs[1], label='EPU+IteRand', legend=False)
sns.lineplot(x=[.5,.95,.9,0.8,.6,.4,.2], y=[88.91,73.5,84.3,87.75,88.64,88.8,88.05] ,ax=axs[1], label='Biprop+Recycle', legend=False)
axs[1].axhline(87, linestyle=':', )
axs[1].set_title(label='Conv-4', fontdict = {'fontsize' : 14})

sns.lineplot(x=[.5,.95,.9,0.8,.6,.4,.2], y=[89.52,66.12,82.38,87.5,90.25,88.9,78.2] ,ax=axs[2],linestyle='-.', label='Biprop', legend=False)
#sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[65.3,58.9,65.72,65,56] ,ax=axs[2], label='Biprop+IteRand', legend=False)
sns.lineplot(x=[.5,.95,.9,0.8,.6,.4,.2], y=[90.9,72.11,85.5,90,90.6,90.8,90.2] ,ax=axs[2], label='Biprop+Recycle', legend=False)
axs[2].axhline(89, linestyle=':', )
axs[2].set_title(label='Conv-6', fontdict = {'fontsize' : 14})

sns.lineplot(x=[.5,.95,.9,0.8,.6,.4,.2], y=[90.35,75.5,84.84,89.11,90.54,89.13,76.59] ,ax=axs[3], label='Biprop', legend=False)
#sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[77.61,76.72,78.9,75.14,49.84] ,ax=axs[3], label='Biprop+IteRand', legend=False)
sns.lineplot(x=[.5,.95,.9,0.8,.6,.4,.2], y=[91,75.5,86.4,90.55,90.85,90.93,90.35] ,ax=axs[3], label='Biprop+Recycle', legend=False)
axs[3].axhline(89.41, linestyle=':', )
axs[3].set_title(label='Conv-8', fontdict = {'fontsize' : 14})


for ax in axs:
    ax.set(xlabel="Percentage of Weights Pruned")
    #ax.get_legend().remove()
axs[0].set(ylabel="CIFAR-10 Test Accuracy")

plt.xlim([.2, .95])
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.12, .02))
#handles, labels = axs[2].get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.62, .02))
plt.tight_layout(rect=[0,.08,1,1])
#print(labels)
sns.set_style('darkgrid')
plt.savefig('figs/conv2to8_biprop_3.pdf')

del fig
del axs





