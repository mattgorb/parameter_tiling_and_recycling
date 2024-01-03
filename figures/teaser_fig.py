import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
#import matplotlib.pyplot as plt


#plt.rcParams["image.cmap"] = "Set1"
# to change default color cycle
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

N=10
cmap = plt.cm.get_cmap('tab10', N)
colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(N)]
print(colors)

color1=colors[0]
color2=colors[1]
#sys.exit()
# Data for the horizontal lines
#models = ["mlpmixer baseline (params 2m)", "convmixup baseline (params 5m)"]
#values = [2, 5]  # Adjust these values based on your actual data

# Create a Matplotlib plot with two horizontal lines
plt.figure(figsize=(8, 5))
fig, ax = plt.subplots()


#for model, value in zip(models, values):
#plt.text(0.5, value, f'{model}\n({value}m)', color='black', ha='right', va='center', backgroundcolor='white', fontsize=8)
#t=plt.text(433088, 79.75, f'Full-Precision ()', color='black', ha='left', va='top', backgroundcolor='white', fontsize=6)
#t.set_bbox(dict( alpha=0,))

#t=plt.text(0.75, 92.5, f'ConvMixer, Full-Precision (1.34M FP-32 Params)', color='black', ha='right', va='top', backgroundcolor='white', fontsize=7)
#t.set_bbox(dict( alpha=0, ))
t=plt.text(70000, 82, f'1.82M 32-Bit Params', color='black', ha='left', va='top', backgroundcolor='white', fontsize=9)
t.set_bbox(dict( alpha=0, ))

t=plt.text(750000, 92, f'1.34M 32-Bit Params', color='black', ha='left', va='top', backgroundcolor='white', fontsize=9)
t.set_bbox(dict( alpha=0, ))

#conv
plt.axhline(y=91.25,  linestyle='dotted', color=color1, label='ConvMixer, Full-Precision')
plt.plot([538688,409664, 348224,322000], [91.3, 89.9,86.8, 84.2], color=color1,label='Tiled ConvMixer' , linestyle='-',marker='o', )

#mlp
plt.axhline(y=81.25, linestyle='-.',color=color2,label='MLPMixer, Full-Precision')
plt.plot([628160, 433088, 337856, 232064], [81.45,80.21,78.8,76.9 ],color=color2 , label='Tiled MLPMixer' , linestyle='--',marker='o', )

t=plt.text(538688+85000, 91.3-.4, r'$\mathrm{TBN_{4}}$', color='black', ha='right', va='top', backgroundcolor='white',fontsize=13)
t.set_bbox(dict( alpha=0, ))

t=plt.text(409664, 89.9-0.4, r'$\mathrm{TBN_{8}}$', color='black', ha='left', va='top', backgroundcolor='white', fontsize=13)
t.set_bbox(dict( alpha=0, ))

t=plt.text(348224, 86.8-0.4, r'$\mathrm{TBN_{16}}$', color='black', ha='left', va='top', backgroundcolor='white', fontsize=13)
t.set_bbox(dict( alpha=0, ))

t=plt.text(322000, 84.2-0.4, r'$\mathrm{TBN_{32}}$', color='black', ha='left', va='top', backgroundcolor='white', fontsize=13)
t.set_bbox(dict( alpha=0, ))


t=plt.text(628160-5000, 81, r'$\mathrm{TBN_{4}}$', color='black', ha='left', va='top', backgroundcolor='white', fontsize=13)
t.set_bbox(dict( alpha=0, ))

t=plt.text(433088-5000, 79.85, r'$\mathrm{TBN_{8}}$', color='black', ha='left', va='top', backgroundcolor='white', fontsize=13)
t.set_bbox(dict( alpha=0, ))

t=plt.text(337856-5000, 78.45, r'$\mathrm{TBN_{16}}$', color='black', ha='left', va='top', backgroundcolor='white', fontsize=13)
t.set_bbox(dict( alpha=0, ))

t=plt.text(232064-5000, 76.55, r'$\mathrm{TBN_{32}}$', color='black', ha='left', va='top', backgroundcolor='white', fontsize=13)
t.set_bbox(dict( alpha=0, ))


# Add labels and title
plt.xlabel("Number of Binary Parameters", fontsize=13)
plt.ylabel("Test Set Accuracy", fontsize=13)
#plt.legend(loc='center right', fontsize=12)
plt.legend(loc='center right',bbox_to_anchor=(0,0,1,1.1), fontsize=12)
plt.ylim(75.5,93)
plt.xlim(20000, 1000000)
plt.tight_layout()


from matplotlib.ticker import FuncFormatter
def millions_formatter(x, pos):
    return f'{x / 1e3:.0f}K'
plt.gca().xaxis.set_major_formatter(FuncFormatter(millions_formatter))
x = [200000, 400000,600000,800000,1000000,]
ax.set_xticks(x, fontsize=12)


#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)

plt.savefig('teaser2.pdf')