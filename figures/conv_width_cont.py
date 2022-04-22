import matplotlib.pyplot as plt
import seaborn as sns


col=3
plt.clf()
sns.set_style('whitegrid')
fig, axs = plt.subplots(1, col, sharex=True, figsize=(5*col,6))
'''sns.lineplot(x=[0.1,0.25,0.5,1], y=[48.1,64.52,73.65,79.69] ,ax=axs[0], linestyle='-.',label='Edge-Popup', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[55.2,70.5,78,81.83] ,ax=axs[0],linestyle='-.', label='IteRand', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[53.4,70.33,77,81.68] ,ax=axs[0], label='Edge-Popup+Recycle', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[58.5,72.49,77.55,79.9] ,ax=axs[0],linestyle='--',label='Baseline (Learned Weights)', legend=False)
axs[0].set_title(label='Wide Conv-2', fontdict = {'fontsize' : 14})'''

sns.lineplot(x=[0.1,0.25,0.5,1], y=[50.73,73.34,82.08,86.67] ,ax=axs[0],linestyle='-.', label='Edge-Popup', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[59.3,78.91,85.45,88.37] ,ax=axs[0],linestyle='-.', label='IteRand', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[59.82,78.92,85.63,88.5] ,ax=axs[0], label='Edge-Popup+Recycle', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[67.26,81.13,84.96,86.66] ,linestyle='--',ax=axs[0], label='Baseline (Learned Weights)', legend=False)
axs[0].set_title(label='Wide Conv-4', fontdict = {'fontsize' : 18})




sns.lineplot(x=[0.1,0.25,0.5,1], y=[57.3,78.48,86.24,89.53] ,ax=axs[1],linestyle='-.', label='Edge-Popup', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[64.5,83,88.18,90.74] ,ax=axs[1],linestyle='-.', label='IteRand', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[64.1,83,88.57,90.9] ,ax=axs[1], label='Edge-Popup+Recycle', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[72.48,83.65,87.54,89.14] ,linestyle='--',ax=axs[1], label='Baseline (Learned Weights)', legend=False)
axs[1].set_title(label='Wide Conv-6', fontdict = {'fontsize' : 18})

sns.lineplot(x=[0.1,0.25,0.5,1], y=[63.35,81.5,87.92,90.28] ,ax=axs[2],linestyle='-.', label='Edge-Popup', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[71.56,85.5,87.99,91.11] ,ax=axs[2],linestyle='-.', label='IteRand', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[70,85.5,89.73,91.36] ,ax=axs[2], label='Edge-Popup+Recycle', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[75.06,84.75,87.18,89.41] ,linestyle='--',ax=axs[2], label='Baseline (Learned Weights)', legend=False,)
#axs[3].title.set_text('Wide Conv-8',)
axs[2].set_title(label='Wide Conv-8', fontdict = {'fontsize' : 18})
#axs[3].axhline(89.41, linestyle=':', )

for ax in axs:
    ax.set_xlabel("Layer Width Factor", fontdict = {'fontsize' : 18})
    #ax.get_legend().remove()
axs[0].set_ylabel("CIFAR-10 Test Accuracy",fontdict = {'fontsize' : 18} )


plt.xticks([.1,.25, .5, 1])
plt.xlim([.1, 1])
#handles, labels = axs[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.12, .02))
handles, labels = axs[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', ncol=4,bbox_to_anchor=(.10, .01), prop={'size': 18})
plt.tight_layout(rect=[0,.1,1,1])

#plt.show()
plt.savefig('width_epu2.pdf')

del fig
del axs





