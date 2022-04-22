import matplotlib.pyplot as plt
import seaborn as sns


col=2
plt.clf()
sns.set_style('whitegrid')
fig, axs = plt.subplots(1, col, sharex=True, figsize=(5*col,5))


sns.lineplot(x=[2150496,430099,860198,1720396,2580595,3440793], y=[81.36,76.9,80.23,81.36,81.2,80.28] ,ax=axs[0],linestyle='-.', label='Conv-2', legend=False)
sns.lineplot(x=[1212512.00,242502.40,485004.80,970009.60,1455014.40,1940019.20], y=[88.91,84.3,87.75,88.64,88.8,88.05] ,ax=axs[0],linestyle='-.', label='Conv-4', legend=False)
sns.lineplot(x=[1130592.00,226118.40,452236.80,904473.60,1356710.40,1808947.20], y=[90.9,85.5,90,90.6,90.8,90.2] ,ax=axs[0], label='Conv-6', legend=False)
sns.lineplot(x=[2637920.00,527584.00,1055168.00,2110336.00,3165504.00,4220672.00], y=[91,86.4,90.55,90.85,90.93,90.35] ,ax=axs[0], label='Conv-8', legend=False)
sns.lineplot(x=[2425024,2261184,5275840,11678912], y=[87, 89, 89.4,93] ,markers='.',linestyle='--',ax=axs[0], label='Baseline (Learned Weights)', legend=False)
axs[0].set_title(label='Wide Conv-4', fontdict = {'fontsize' : 18})




'''sns.lineplot(x=[0.1,0.25,0.5,1], y=[57.3,78.48,86.24,89.53] ,ax=axs[1],linestyle='-.', label='Edge-Popup', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[64.5,83,88.18,90.74] ,ax=axs[1],linestyle='-.', label='IteRand', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[64.1,83,88.57,90.9] ,ax=axs[1], label='Edge-Popup+Recycle', legend=False)
sns.lineplot(x=[0.1,0.25,0.5,1], y=[72.48,83.65,87.54,89.14] ,linestyle='--',ax=axs[1], label='Baseline (Learned Weights)', legend=False)
axs[1].set_title(label='Wide Conv-6', fontdict = {'fontsize' : 18})'''



for ax in axs:
    ax.set_xlabel("Number of Parameters", fontdict = {'fontsize' : 18})
    #ax.get_legend().remove()
axs[0].set_ylabel("CIFAR-10 Test Accuracy",fontdict = {'fontsize' : 18} )


#plt.xticks([.1,.25, .5, 1])
#plt.xlim([.1, 1])
#handles, labels = axs[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.12, .02))
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', ncol=4,bbox_to_anchor=(.10, .01), prop={'size': 18})
plt.tight_layout(rect=[0,.1,1,1])

#plt.show()
plt.savefig('figs/params_small.pdf')

del fig
del axs





