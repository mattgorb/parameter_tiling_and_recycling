import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# data from https://allisonhorst.github.io/palmerpenguins/

species = (
    'ResNet18',
    'ResNet50',
    'PointNet',
    "MLPMixer",
    "ViT(Small)",
    "Swin-t",
    'Mobile ViT'

)
weight_counts = {
    "Fully-Connected": np.array([0,	0, 3027200	,1602560,	9466880	,26550912, 1557504]),#, 1177344]),
    "Conv. Kernel Size > 1x1": np.array([10987200,11318976,	0,	0,	0,	0,544752]),# 544752]),#]),
    "Conv. Kernel Size = 1x1": np.array([177152,12148736,	422272,	196608,	0,	0,260864]),# 260864]),
    #"Other": np.array([9600	,53120, 22001	,20362,	23050,	47146,19824]),
}


'''
df = pd.DataFrame(weight_counts,
                  species)
 
 
# create stacked bar chart for monthly temperatures
df.plot(kind='bar', stacked=False, width=0.3 )#color=['red', 'skyblue', 'green'])
 
# labels for x & y axis
plt.xlabel('Months')
plt.ylabel('Temp ranges in Degree Celsius')
 
# title of plot
plt.title('Monthly Temperatures in a year')

plt.savefig('test.png')
'''

import matplotlib.pyplot as plt
# to change default colormap
plt.rcParams["image.cmap"] = "Dark2"
# to change default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)

width = 0.5
fig, ax = plt.subplots(figsize=(28,10))
#fig, ax = plt.subplots()
bottom = np.zeros(7)
plt.rcParams['font.size'] = 18


patterns = [None, "/" , "\\"]
i=0
ps=[]
for boolean, weight_count in weight_counts.items():
    p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom,hatch=patterns[i] )
    bottom += weight_count
    i+=1
    

#ax.set_title("Proportion of Parameter Types in Different Architectures", fontsize=40)
plt.ylabel('Number of Parameters', fontsize=45)
#plt.xlabel('Architecture', fontsize=20)
#ax.legend(loc='best', fontsize=26)
ax.legend(loc="upper center", fontsize=36)

plt.xticks( fontsize=46,)# fontname='monospace')

from matplotlib.ticker import FuncFormatter
def millions_formatter(x, pos):
    return f'{x / 1e6:.0f}M'
plt.gca().yaxis.set_major_formatter(FuncFormatter(millions_formatter))

y = [10000000, 20000000,]
def millions_formatter(x, pos):
    return f'{int(x / 1e6)}M'

# Apply the custom formatter to y-axis ticks
ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

# Set fontsize for y-axis tick labels
ax.tick_params(axis='y', labelsize=48)
#ax.set_yticks(y, fontsize=40)
#ax.set_yticklabels(y, fontsize=40)
#for tick in plt.gca().yaxis.get_major_ticks():
    #tick.label.set_fontsize(34) 



# Hide the top border by setting the color of the top side of the bars to match the background color
# Set edge color for the last axis
ax.spines['top'].set_edgecolor('white')

plt.tight_layout()
plt.savefig('barchart_new.pdf')
''''''