import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from seaborn import lineplot,scatterplot

jaccard=True
prune_rate=None
if jaccard:
    #col='jaccard_total'
    savefile=f'tables/masks_graph_jaccard_{prune_rate}.pdf'
    cols=['model_combo','convs.0_jaccard','convs.2_jaccard','convs.5_jaccard','convs.7_jaccard'
          ,'convs.10_jaccard','convs.12_jaccard','linear.0_jaccard','linear.2_jaccard','linear.4_jaccard']
else:
    #col='percent_same_total'
    savefile=f'tables/masks_graph_pct_same_{prune_rate}.pdf'
    cols=['model_combo','convs.0_pctsame','convs.2_pctsame','convs.5_pctsame','convs.7_pctsame'
          ,'convs.10_pctsame','convs.12_pctsame','linear.0_pctsame','linear.2_pctsame','linear.4_pctsame']

fig = plt.figure(figsize=(8,8))
sns.set_style('darkgrid')
df1=pd.read_csv(f'tables/model_combos_conv6-sc-epu-var_{prune_rate}.csv')
print(df1.columns)
df1=df1[cols]
print(df1.shape)
dfm = df1.melt('model_combo', var_name='cols', value_name='vals')
print(dfm.shape)
dfm2=dfm.groupby('cols', as_index=False).agg({
'vals': ['mean', 'min', 'max']
 })
dfm2.columns = dfm2.columns.droplevel(0)#.reset_index()
dfm2 = dfm2.rename(columns = {"" : "cols"})
print(dfm2)



df1=pd.read_csv(f'tables/model_combos_conv6-kn-biprop-var_{prune_rate}.csv')
print(df1.columns)
df1=df1[cols]
print(df1.shape)
dfm = df1.melt('model_combo', var_name='cols', value_name='vals')
print(dfm.shape)
dfm3=dfm.groupby('cols', as_index=False).agg({
'vals': ['mean', 'min', 'max']
 })
dfm3.columns = dfm3.columns.droplevel(0)#.reset_index()
dfm3 = dfm3.rename(columns = {"" : "cols"})

#ax=scatterplot(dfm2['cols'], dfm2['mean'], s=50, )
ax =lineplot(data=dfm2, x='cols', y="mean", marker="8",markersize=7,label='Edge-Popup')
ax.fill_between(dfm2['cols'], dfm2['min'], dfm2['max'], alpha=0.2)
ax =lineplot(data=dfm3, x='cols', y="mean", marker="v",markersize=7,label='Biprop')
ax.fill_between(dfm3['cols'], dfm3['min'], dfm3['max'], alpha=0.2)
ax.set(xlim=(-.05, 8.05))
#ax.set(xlabel='Model Layer', size=12)

ax.set_xlabel('Model Layer', fontsize=16)
ax.set(xticklabels=['Conv. 1','Conv. 2','Conv. 3','Conv. 4','Conv. 5','Conv. 6','Lin. 1','Lin. 2', 'Lin.3'])
#ax.set(yticklabels=['','50','55','60','65','70','75'])
plt.xticks(rotation=30, size = 12)
print(ax.get_xlim())
if jaccard:
    ax.set_ylabel('Jaccard Index', fontsize=16)
else:
    yticks=ax.get_yticks()
    yticks=[f"{round(i*100,2)}%" for i in yticks]
    ax.set(yticklabels=yticks)
    ax.set_ylabel('Percent Matching', fontsize=16)
plt.legend(fontsize='x-large',)
plt.tight_layout()
plt.savefig(savefile)
