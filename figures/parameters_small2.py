import matplotlib.pyplot as plt
import seaborn as sns


col=1
plt.clf()
sns.set_style('whitegrid')
fig, axs = plt.subplots(1, col, sharex=True, figsize=(7*col,5))

'''
sns.lineplot(x=[2150496,430099,860198,1720396,2580595,3440793], y=[81.36,76.9,80.23,81.36,81.2,80.28] ,ax=axs[0],linestyle='-.', label='Conv-2', legend=False)
sns.lineplot(x=[1212512.00,242502.40,485004.80,970009.60,1455014.40,1940019.20], y=[88.91,84.3,87.75,88.64,88.8,88.05] ,ax=axs[0],linestyle='-.', label='Conv-4', legend=False)
sns.lineplot(x=[1130592.00,226118.40,452236.80,904473.60,1356710.40,1808947.20], y=[90.9,85.5,90,90.6,90.8,90.2] ,ax=axs[0], label='Conv-6', legend=False)
sns.lineplot(x=[2637920.00,527584.00,1055168.00,2110336.00,3165504.00,4220672.00], y=[91,86.4,90.55,90.85,90.93,90.35] ,ax=axs[0], label='Conv-8', legend=False)
sns.lineplot(x=[2425024,2261184,5275840,11678912], y=[87, 89, 89.4,93] ,markers='.',linestyle='--',ax=axs[0], label='Baseline (Learned Weights)', legend=False)
'''

#sns.lineplot(x=[19881	134808	538160	2150496], y=[] ,ax=axs[0], label='Conv-2-Widths (EPU)', legend=False)
#sns.lineplot(x=[11253	76184	303664	1212512], y=[] ,ax=axs[0], label='Conv-4-Widths', legend=False)
#sns.lineplot(x=[10815	71064	283184	1130592], y=[] ,ax=axs[0], label='Conv-6-Widths', legend=False)
#sns.lineplot(x=[25807	165016	659696	2637920], y=[] ,ax=axs[0], label='Conv-8-Widths', legend=False)

#sns.lineplot(x=[19881,134808,538160,2150496], y=[52.55,68.8,76.66,81.36] ,ax=axs[0], label='Conv-2-Widths', legend=False)
#sns.lineplot(x=[76184,303664,1212512], y=[78.51,86.5,88.91] ,ax=axs[0], label='Conv-4-Widths', legend=False)
#sns.lineplot(x=[71064,283184,1130592], y=[82,88.43,90.9] ,ax=axs[0], label='Conv-6-Widths', legend=False)
#sns.lineplot(x=[25807,165016,659696,2637920], y=[69,84.51,89.67,91] ,ax=axs[0], label='Conv-8-Widths', legend=False)

#sns.lineplot(x=[39761,269616,1076320,4300992], y=[58.5,72.49,77.55,79.9] ,ax=axs[0], label='Conv-2-Widths (Baselines)', legend=False)
#ns.lineplot(x=[152368,607328,2425024], y=[81.13,84.96,86.66] ,ax=axs[0], label='Conv-4-Widths (Baselines)', legend=False)
#sns.lineplot(x=[142128,566368,2261184], y=[83.65,87.54,89.14] ,ax=axs[0], label='Conv-6-Widths (Baselines)', legend=False)
#sns.lineplot(x=[51614,330032,1319392,5275840], y=[75.06,84.75,87.18,89.41] ,ax=axs[0], label='Conv-8-Widths (Baselines)', legend=False)

axs.errorbar([48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80],
             [49.38,72.11,73.5,85.5,84.3,76.9,90,87.75],[1.23,1.74,1.55,1.37,1.34,1.18, 1.12, 1.] , label='Weight Recycle', )

sns.lineplot(x=[48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80],
             y=[41.6,68.8,64.72,81.22,79.78,68.87,86.94,85.34] ,ax=axs, label='IteRand',marker="o", markersize=4, legend=False)
sns.lineplot(x=[48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80],
             y=[35.28,60.63,57.61,79.57,74.42,64.05,85.4,83.28] ,ax=axs, label='Edge-Popup', marker="v", markersize=6,linestyle='-.',legend=False)
sns.lineplot(x=[48500.48,113059.20,121251.20,226118.40,242502.40,430099.20,452236.80,485004.80],
             y=[44,66.12,64.83,82.38,79,70.1,87.5,85.34] ,ax=axs,linestyle='-.',marker="^", markersize=6, label='Biprop', legend=False)

axs.text(48500.48,33,"Conv-4, 98% Pruned (2.4M)",weight='bold',fontsize=8,)
#axs.text(75500.48,75,"Conv-6, 95% \nPruned (2.2M)",weight='bold',fontsize=8,)
axs.text(121251.48,55,"Conv-4, 95% (2.4M)",weight='bold',fontsize=8,)
axs.text(226118.48,87.75,"Conv-2, 95% (4.3M)",weight='bold',fontsize=8,)
axs.text(245502.48,84.9,"Conv-6, 90% (2.2M)",weight='bold',fontsize=8,)
axs.text(415236.48,91.5,"Conv-6, 80% (2.2M)",weight='bold',fontsize=8,)


#for ax in axs:
axs.set_xlabel("Number of Parameters (Thousands)", fontdict = {'fontsize' : 15})
    #ax.get_legend().remove()
axs.set_ylabel("CIFAR-10 Test Accuracy",fontdict = {'fontsize' : 15} )

axs.set(xticklabels=["",100,200,300,400,500])

#plt.xticks()
plt.ylim([25,95])
#handles, labels = axs[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower left', ncol=3,bbox_to_anchor=(.12, .02))
handles, labels = axs.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', ncol=4,bbox_to_anchor=(.02, .01), prop={'size': 11})
plt.tight_layout(rect=[0,.08,1,1])

#plt.show()
plt.savefig('figs/params_small2.pdf')

del fig
del axs





