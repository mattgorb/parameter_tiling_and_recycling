import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist,squareform
import numpy as np
from matplotlib.colors import ListedColormap

jaccard=True
prune_rate=0.5
if jaccard:
    col='jaccard_total'
    savefile=f'tables/heatmap_jaccard_{prune_rate}.pdf'
else:
    col='percent_same_total'
    savefile=f'tables/heatmap_pctsame_{prune_rate}.pdf'

df1=pd.read_csv(f'tables/model_combos_conv6-sc-epu-var_{prune_rate}.csv')
df1_pctsame=df1[col].values
min_val1=min(df1_pctsame)
max_val1=max(df1_pctsame)
avg_val1=np.mean(np.array(df1_pctsame))


df1=pd.read_csv(f'tables/model_combos_conv6-kn-biprop-var_{prune_rate}.csv')
df1_pctsame2=df1[col].values
#print(df1_pctsame.shape)
df1_pctsame2=np.array(df1_pctsame2)
df1_pctsame2=np.array(df1_pctsame2)
min_val2=min(df1_pctsame2)
max_val2=max(df1_pctsame2)
avg_val2=np.mean(np.array(df1_pctsame2))

df1_pctsame=squareform(df1_pctsame)
df1_pctsame2=squareform(df1_pctsame2)
result=np.array(np.tril(df1_pctsame)+np.triu(df1_pctsame2))

f, ax = plt.subplots(figsize=(12, 12))

cmap=sns.color_palette('Blues',)
cmap=sns.blend_palette(cmap, n_colors=16)
cmap=cmap[:10]

cmap1=cmap[:int(len(cmap)/2)]
cmap1 = ListedColormap(cmap1)

cmap2=cmap[int(len(cmap)/2):]
cmap2 = ListedColormap(cmap2)
x=np.tril(df1_pctsame2)


cmap_args={'use_gridspec':True,"shrink": .72, "location":"bottom",'pad':0.07,'anchor':(0.15, 1.0)}


sns.heatmap(np.tril(df1_pctsame),   cmap=cmap1,mask=np.triu(df1_pctsame),vmin=min_val1,axes=ax, vmax=max_val1, center=avg_val1,
            square=False, cbar_kws=cmap_args)
sns.heatmap(np.triu(df1_pctsame2),  cmap=cmap2,mask=np.tril(df1_pctsame2), vmin=min_val2,vmax=max_val2, axes=ax,center=avg_val2 ,
            square=False, cbar_kws={'use_gridspec':True,"shrink": .85 })
sns.heatmap(result ,mask=result, cbar=False,cmap=ListedColormap(['lavender']),vmin=0,vmax=0,square=False,linewidths=.004, )


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

cbar.set_ticks([min_val1,  max_val1])
if not jaccard:
    #cbar.set_label("% Matching Masks",size=16)
    cbar.set_ticklabels([f'{round(min_val1*100,3)}%' ,f'{round(max_val1*100,3)}%'])
else:
    cbar.set_ticklabels([f'{round(min_val1 , 4)}', f'{round(max_val1 , 4)}'])
cbar = ax.collections[1].colorbar
cbar.ax.tick_params(labelsize=18)
cbar.set_ticks([min_val2,  max_val2])
if not jaccard:
    #cbar.set_label("% Matching Masks, test", size=16)
    cbar.set_ticklabels([f'{round(min_val2*100,3)}%',f'{round(max_val2*100,3)}%'])
else:
    cbar.set_ticklabels([f'{round(min_val2,4)}',f'{round(max_val2,4)}'])
ax.set_title('Biprop', fontsize=34)
ax.set_xlabel('Score Parameter Seed Number', fontsize=22)
ax.set_ylabel('Edge-Popup', fontsize=34)
#ax.set(xlabel='Score Parameter Seed Number\ncommon xlabel', ylabel='Edge-Popup\nScore Parameter Seed Number', fontsize=18)
#f.set_axis_labels(None, 'Values')
plt.tight_layout(rect=[0,-.08,1,1])
#plt.tight_layout()#rect=[0,.1,1,1]
print(savefile)
plt.savefig(savefile)