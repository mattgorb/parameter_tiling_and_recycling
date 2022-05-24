import matplotlib.pyplot as plt
import seaborn as sns


col=4
plt.clf()
sns.set_style('whitegrid')

x=sns.color_palette("light:b",  5)
print(x.as_hex())

#sns.set_palette(sns.dark_palette("#69d",  as_cmap=True))
sns.lineplot(x=[i for i in range(11)], y=[12.83438969,13.88095951,20.7807312,22.25767326,32.5100708,33.78979111,49.41247177,50.68508148,30.90458298,25.45535469,4.678293705] ,linestyle='-.',color='#b4b4f6',linewidth=1,label='Biprop+Recycle+', legend=False)
sns.lineplot(x=[i for i in range(11)], y=[13.07181358,14.00544453,20.8949585,22.25463676,32.55274582,33.61766052,49.23002243,50.33876038,30.44027901,24.18491745,4.774709702] , linestyle='-.',color='#7878f9',linewidth=1,label='Biprop+Recycle-', legend=False)




#x=sns.light_palette("seagreen",  5)
#print(x.as_hex())
#sns.lineplot(x=[i for i in range(11)],y=[12.3318758,14.76031876,19.13018417,18.97434044,23.01779175,24.32123947,26.66713333,24.82845306,15.9275465,15.07482719,6.408082485] ,linestyle='-',color='#2e8b57',linewidth=1.5,label='Biprop + Recycle +', legend=False)
sns.lineplot(x=[i for i in range(11)],y=[12.53140831,11.84542179,17.02272034,17.38848686,25.39003563,25.67652321,37.25827026,37.68733597,25.2218132,23.56786156,4.639759064] , linestyle='-',color='#5da57c',linewidth=1.5,label='Edge-Popup+', legend=False)
sns.lineplot(x=[i for i in range(11)], y=[11.60200214,11.38171005,15.95085049,15.99358845,22.6307888,22.64547539,31.9877739,31.98503494,22.62521935,22.54005623,4.48478651] , linestyle='dotted',color='#8cbfa2',linewidth=1.25,label='Edge-Popup-', legend=False)

x=sns.color_palette("flare",  6,)
print(x.as_hex()),sns.lineplot(x=[i for i in range(11)], y=[14.31690979,13.78635311,19.84573746,21.37153625,30.61770439,31.2269249,44.85501862,45.92207336,29.59231186,25.18566895,5.150443554] , linestyle='dashed',color='indianred',linewidth=1,label='IteRand +', legend=False)
sns.lineplot(x=[i for i in range(11)],y=[13.24188805,12.56322956,18.13490105,18.88751984,26.87413597,27.28631592,39.05298233,39.60852432,26.30578804,23.94405937,4.821712017] , linestyle='dotted',color='red',linewidth=1,label='IteRand -', legend=False)
sns.lineplot(x=[i for i in range(11)],y=[13.04279995,12.36833858,18.02428055,18.28163528,26.41761017,26.57927322,38.17338943,38.50655746,26.58241653,24.72285652,5.195301533] , linestyle='dashed',color='darkorchid',linewidth=1,label='Biprop +', legend=False)
sns.lineplot(x=[i for i in range(11)],y=[11.81847191,11.33716965,16.00897789,15.98553562,22.62303925,22.652071,32.001194,31.97277451,22.65958786,22.45717812,4.539419651] , linestyle='dotted',color='indigo',linewidth=1,label='Biprop -', legend=False)


#g.set(xticklabels=['Conv. 1','Conv. 2','Conv. 3','Conv. 4','Conv. 5','Conv. 6','Conv. 7','Conv. 8','Lin. 1','Lin. 2', 'Lin.3'])
#sg.set_xticklabels(['Conv. 1','Conv. 2','Conv. 3','Conv. 4','Conv. 5','Conv. 6','Conv. 7','Conv. 8','Lin. 1','Lin. 2', 'Lin.3'])
plt.xticks([i for i in range(11)], ['Conv. 1','Conv. 2','Conv. 3','Conv. 4','Conv. 5','Conv. 6','Conv. 7','Conv. 8','Lin. 1','Lin. 2', 'Lin.3'],rotation=30,)
#plt.title('Norms at')
#plt.legend()
plt.ylabel('Norm', fontsize=12)
plt.xlabel('Layer', fontsize=12)
plt.legend(bbox_to_anchor=(-.08, 1.19), loc=2, ncol=3)
#plt.tight_layout(rect=[2,2,2,2])
plt.savefig('figs/norms2.pdf', bbox_inches='tight')