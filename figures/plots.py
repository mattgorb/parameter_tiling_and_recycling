import matplotlib.pyplot as plt
import seaborn as sns


plt.clf()
fig, axs = plt.subplots(1, 4, sharex=True, figsize=(20,5))
sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[79.69,75.15,79.1,78.6,70.06] ,ax=axs[0], legend=False,label='1')
sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[79.69,75.15,79.1,78.6,70.06],ax=axs[1], legend=False,label='2')
sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[79.69,75.15,79.1,78.6,70.06], ax=axs[2], legend=False,label='3')
sns.lineplot(x=[.5,0.8,.6,.4,.2], y=[79.69,75.15,79.1,78.6,70.06], ax=axs[3], legend=False,label='4')
for ax in axs:
    ax.set(xlabel="Percentage of Weights Pruned")
    #ax.legend=False
axs[0].set(ylabel="CIFAR-10 Test Accuracy")
sns.set_style('darkgrid')
#handles1, labels1 = axs[0].get_legend_handles_labels()
#handles2, labels2 = axs[1].get_legend_handles_labels()
#fig.legend(False)
axs[0].legend(title='Smoker', loc='upper left', labels=['Hell Yeh', 'Nah Bruh'])
#plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()
