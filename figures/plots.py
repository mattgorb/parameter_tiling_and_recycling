import matplotlib.pyplot as plt
import seaborn as sns


col=4
plt.clf()
fig, axs = plt.subplots(1, col, sharex=True, figsize=(5*4,5))
sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[79.69,75.15,79.1,78.6,70.06] ,ax=axs[0], label='Edge-Popup (EPU)', legend=False)
sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[81.83,78.04,81.75,81.3,80.7] ,ax=axs[0], label='EPU+IteRand', legend=False)
sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[81.68,79.06,81.12,81.8,81.05] ,ax=axs[0], label='EPU+Recycle', legend=False)
axs[0].axhline(79.9, linestyle=':', )
axs[0].title.set_text('Conv2')

sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[90.28,86.9,90.5,90.15,84.39] ,ax=axs[1], label='Edge-Popup (EPU)', legend=False)
sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[91.11,88.42,91.16,91.55,90.99] ,ax=axs[1], label='EPU+IteRand', legend=False)
sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[91.36,88.87,90.95,91.5,91.15] ,ax=axs[1], label='EPU+Recycle', legend=False)
axs[1].axhline(89.41, linestyle=':', )
axs[1].title.set_text('Conv8')




sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[79.56,77.9,79.98,78.3,64.1] ,ax=axs[2], label='Biprop', legend=False)
#sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[65.3,58.9,65.72,65,56] ,ax=axs[2], label='Biprop+IteRand', legend=False)
sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[81.36,80.23,81.36,81.2,80.28] ,ax=axs[2], label='Biprop+Recycle', legend=False)
axs[2].axhline(79.9, linestyle=':', )
axs[2].title.set_text('Conv2')

sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[90.35,89.11,90.54,89.13,76.59] ,ax=axs[3], label='Biprop', legend=False)
#sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[77.61,76.72,78.9,75.14,49.84] ,ax=axs[3], label='Biprop+IteRand', legend=False)
sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[91,90.55,90.85,90.93,90.35] ,ax=axs[3], label='Biprop+Recycle', legend=False)
axs[3].axhline(89.41, linestyle=':', )
axs[3].title.set_text('Conv8')


for ax in axs:
    ax.set(xlabel="Percentage of Weights Pruned")
    #ax.get_legend().remove()
axs[0].set(ylabel="CIFAR-10 Test Accuracy")
sns.set_style('darkgrid')
plt.xlim([.2, .8])
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.12, .02))
handles, labels = axs[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.62, .02))
plt.tight_layout(rect=[0,.08,1,1])
print(labels)
plt.savefig('figures/test.png')

del fig
del axs





