import matplotlib.pyplot as plt
import seaborn as sns


col=1
plt.clf()
sns.set_style('whitegrid')
fig, axs = plt.subplots(1, col, sharex=True, figsize=(5*col,5))

'''
axs.errorbar([48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80,],
             [49.38,72.11,73.5,85.5,84.3,76.9,90,87.75,],[.87,.74,.55,.37,.34,.98, 1.12, .79,] , label='Weight Recycle', )
sns.lineplot(x=[48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80,],
             y=[41.6,68.8,64.72,81.22,79.78,68.87,86.94,85.34,] ,ax=axs, label='IteRand',marker="o", markersize=4, legend=False)
sns.lineplot(x=[48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80,],
             y=[35.28,60.63,57.61,79.57,74.42,64.05,85.4,83.28,] ,ax=axs, label='Edge-Popup', marker="v", markersize=6,linestyle='-.',legend=False)
sns.lineplot(x=[48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80,],
             y=[44,66.12,64.83,82.38,79,70.1,87.5,85.34,] ,ax=axs,linestyle='-.',marker="^", markersize=6, label='Biprop', legend=False)

axs.text(48500.48,33,"Conv-4, 98% Pruned (2.4M)",weight='bold',fontsize=8,)
axs.text(121251.48,55,"Conv-4, 95% (2.4M)",weight='bold',fontsize=8,)
axs.text(226118.48,87.75,"Conv-2, 95% (4.3M)",weight='bold',fontsize=8,)
axs.text(245502.48,84.9,"Conv-6, 90% (2.2M)",weight='bold',fontsize=8,)
axs.text(415236.48,91.5,"Conv-6, 80% (2.2M)",weight='bold',fontsize=8,)
'''

axs.errorbar([116789.12,226118.40,583945.60],
             [85.3,90.9,94.16],[0,0,0] , label='Weight Recycle', )
sns.lineplot(x=[116789.12,226118.40,583945.60],
             y=[80.25,84.2,86.52] ,ax=axs, label='IteRand',marker="o", markersize=4, legend=False)
sns.lineplot(x=[116789.12,226118.40,583945.60],
             y=[81.67,89.85,92.5] ,ax=axs, label='Edge-Popup', marker="v", markersize=6,linestyle='-.',legend=False)
sns.lineplot(x=[116789.12,226118.40,583945.60],
             y=[82.6,89.57,92.62] ,ax=axs,linestyle='-.',marker="^", markersize=6, label='Biprop', legend=False)

axs[2].axhline(89, linestyle=':', linewidth=2.5, color='c', )
#for ax in axs:
axs.set_xlabel("Number of Parameters (Thousands)", fontdict = {'fontsize' : 15})
    #ax.get_legend().remove()
axs.set_ylabel("CIFAR-10 Test Accuracy",fontdict = {'fontsize' : 15} )

axs.set(xticklabels=["",100,200,300,400,500])

#plt.xticks()
plt.ylim([80,95])
#handles, labels = axs[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.12, .02))
handles, labels = axs.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', ncol=4,bbox_to_anchor=(.02, .01), prop={'size': 11})
plt.tight_layout(rect=[0,.08,1,1])

#plt.show()
plt.savefig('figs/params_resnet.pdf')

del fig
del axs





