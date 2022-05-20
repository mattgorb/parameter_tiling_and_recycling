import matplotlib.pyplot as plt
import seaborn as sns


col=2
plt.clf()
sns.set_style('whitegrid')
from matplotlib import gridspec
fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])



ax0.errorbar([48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80,],
             [49.38,72.11,73.5,85.5,84.3,76.9,90,87.75,],[.87,.74,.55,.37,.34,.98, 1.12, .79,] , label='Biprop+Recycle', )
sns.lineplot(x=[48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80,],
             y=[41.6,68.8,64.72,81.22,79.78,68.87,86.94,85.34,] ,ax=ax0, label='IteRand',marker="o", markersize=4, legend=False)
sns.lineplot(x=[48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80,],
             y=[35.28,60.63,57.61,79.57,74.42,64.05,85.4,83.28,] ,ax=ax0, label='Edge-Popup', marker="v", markersize=6,linestyle='-.',legend=False)
sns.lineplot(x=[48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80,],
             y=[44,66.12,64.83,82.38,79,70.1,87.5,85.34,] ,ax=ax0,linestyle='-.',marker="^", markersize=6, label='Biprop', legend=False)

ax0.text(48500.48,33,"Conv-4, 98% Prune Rate (2.4M)",weight='bold',fontsize=9,)
ax0.text(121251.48,55,"Conv-4, 95% (2.4M)",weight='bold',fontsize=9,)
ax0.text(226118.48,87.75,"Conv-2, 95% (4.3M)",weight='bold',fontsize=9,)
ax0.text(245502.48,84.9,"Conv-6, 90% (2.2M)",weight='bold',fontsize=9,)
ax0.text(415236.48,91.5,"Conv-6, 80% (2.2M)",weight='bold',fontsize=9,)


ax1.errorbar([116789.12,226118.40,583945.60],
             [85.3,90.9,93.46],[0.8,0.75,0.7] , label='Biprop+Recycle', )
sns.lineplot(x=[116789.12,226118.40,583945.60],
             y=[80.25,84.2,86.52] ,ax=ax1, label='IteRand',marker="o", markersize=4, legend=False)
sns.lineplot(x=[116789.12,226118.40,583945.60],
             y=[81.67,89.85,92.5] ,ax=ax1, label='Edge-Popup', marker="v", markersize=6,linestyle='-.',legend=False)
sns.lineplot(x=[116789.12,226118.40,583945.60],
             y=[82.6,89.57,92.62] ,ax=ax1,linestyle='-.',marker="^", markersize=6, label='Biprop', legend=False)
ax1.text(126789.48,80,"99% P.R.",weight='bold',fontsize=9,)
ax1.text(163118.48,91.1,"98% P.R.",weight='bold',fontsize=9,)
ax1.text(513945.48,93.65,"95% P.R.",weight='bold',fontsize=9,)
ax1.axhline(93.03, linestyle=':', linewidth=2.5, color='c',label='Baseline' )

#for ax in axs:
ax0.set_xlabel("Number of Parameters (Thousands)", fontdict = {'fontsize' : 14})
ax1.set_xlabel("Number of Parameters (Thousands)", fontdict = {'fontsize' : 14})
ax0.set_ylabel("CIFAR-10 Test Accuracy",fontdict = {'fontsize' : 14} )
ax0.set(xticklabels=["",100,200,300,400,500])
ax1.set(xticklabels=["",100,200,300,400,500,600])

ax1.set_title('ResNet18', fontdict={'fontsize': 14})
#ax1.set(title="ResNet18",)
ax0.set_title('VGG', fontdict={'fontsize': 14})

#ax1.set(titlesize=20)

plt.ylim([78,95])
handles, labels = ax1.get_legend_handles_labels()

labels=[labels[4],labels[0],labels[1],labels[2],labels[3]]
handles=[handles[4],handles[0],handles[1],handles[2],handles[3]]
fig.legend(handles, labels, loc='lower left', ncol=5,bbox_to_anchor=(.14, .01), prop={'size': 14})
plt.tight_layout(rect=[0,.08,1,1])

#plt.show()
plt.savefig('figs/params_resnet.pdf')

del fig






