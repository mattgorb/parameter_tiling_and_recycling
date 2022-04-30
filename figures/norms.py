import matplotlib.pyplot as plt
import seaborn as sns


col=4
plt.clf()
sns.set_style('whitegrid')

sns.lineplot(x=[i for i in range(11)], y=[13.04,12.37,18.02,18.28,26.42,26.58,38.17,38.51,26.58,24.72,5.20] ,linestyle='-',label='Biprop Unpruned', legend=False)
sns.lineplot(x=[i for i in range(11)], y=[12.83,13.88,20.78,22.26,32.51,33.79,49.41,50.69,30.90,25.46,4.68] , linestyle='-',label='Recycle Unpruned', legend=False)
sns.lineplot(x=[i for i in range(11)], y=[15.86,15.43,21.79,21.80,30.84,30.84,43.61,43.62,30.85,30.79,6.15] , linestyle='-',label='Base Weights (Highest 50%)', legend=False)
#sns.lineplot(x=[i for i in range(11)], y=[11.19,12.54,16.31,16.87,21.43,22.20,28.78,29.36,19.80,18.51,5.99] , linestyle='-.',label='Dense Trained+', legend=False)


sns.lineplot(x=[i for i in range(11)],y=[11.82,11.34,16.01,15.99,22.62,22.65,32.00,31.97,22.66,22.46,4.54] ,linestyle='-.',label='Biprop Pruned', legend=False)
sns.lineplot(x=[i for i in range(11)],y=[13.07,14.01,20.89,22.25,32.55,33.62,49.23,50.34,30.44,24.18,4.77] , linestyle='-.',label='Recycle Pruned', legend=False)
sns.lineplot(x=[i for i in range(11)],y=[4.23,4.30,6.06,6.05,8.55,8.56,12.07,12.07,8.55,8.58,1.68] , linestyle='-.',label='Base Weights (Lowest 50%)', legend=False)
#sns.lineplot(x=[i for i in range(11)],y=[3.00,3.09,4.12,4.48,5.72,5.77,7.76,7.96,5.37,5.12,1.45] , linestyle='-.',label='Dense Trained-', legend=False)


plt.legend()
plt.show()