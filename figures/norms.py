import matplotlib.pyplot as plt
import seaborn as sns


col=4
plt.clf()
sns.set_style('whitegrid')

x=sns.color_palette("light:b",  5)
print(x.as_hex())

#sns.set_palette(sns.dark_palette("#69d",  as_cmap=True))
sns.lineplot(x=[i for i in range(11)], y=[11.58323765,13.01672363,16.74621964,17.33502007,22.18697739,23.13245964,29.8127594,30.12952805,20.24824524,19.14601517,5.95474577] ,linestyle='-.',color='#b4b4f6',linewidth=1,label='Dense', legend=False)
sns.lineplot(x=[i for i in range(11)], y=[11.20018959,12.61888981,16.21934891,16.76558495,21.43369102,22.37206841,28.77757072,29.07326889,19.53909683,18.44957352,5.787878513] , linestyle='-.',color='#7878f9',linewidth=1,label='Dense L1 Pruned', legend=False)



x=sns.light_palette("seagreen",  3)
print(x.as_hex())
sns.lineplot(x=[i for i in range(11)],y=[12.3318758,14.76031876,19.13018417,18.97434044,23.01779175,24.32123947,26.66713333,24.82845306,15.9275465,15.07482719,6.408082485] ,linestyle='-',color='#8cbfa2',linewidth=1.5,label='Biprop + Recycle', legend=False)
sns.lineplot(x=[i for i in range(11)],y=[12.18790913,14.16650963,17.96160316,18.82012939,22.5821209,23.91002655,27.541996,26.30025673,16.13414001,14.49637699,6.302339077] , linestyle='-',color='#2e8b57',linewidth=1.5,label='Edge-Popup + Recycle', legend=False)




x=sns.color_palette("dark:salmon_r",  8)
print(x.as_hex())
sns.lineplot(x=[i for i in range(11)], y=[10.67497253,11.25951767,14.27645016,14.79872513,18.32226753,18.97887802,22.9484272,22.41898346,15.11315918,14.20545959,5.351770401] , linestyle='dotted',color='#fa8072',linewidth=1,label='Biprop', legend=False)
sns.lineplot(x=[i for i in range(11)],y=[10.68748951,11.16692448,14.3333149,15.01703358,18.53663254,19.2468605,23.85181427,23.54463005,15.48874569,14.26048374,5.36016798] , linestyle='dotted',color='#dd7367',linewidth=1,label='Edge-Popup', legend=False)
sns.lineplot(x=[i for i in range(11)],y=[10.68517303,10.79680157,13.84731579,13.86627769,17.40482712,17.45000267,22.02930832,21.29385948,14.83121967,14.29030991,5.329949856] , linestyle='dotted',color='#c0655b',linewidth=1,label='Edge-Popup + IteRand', legend=False)


#g.set(xticklabels=['Conv. 1','Conv. 2','Conv. 3','Conv. 4','Conv. 5','Conv. 6','Conv. 7','Conv. 8','Lin. 1','Lin. 2', 'Lin.3'])
#sg.set_xticklabels(['Conv. 1','Conv. 2','Conv. 3','Conv. 4','Conv. 5','Conv. 6','Conv. 7','Conv. 8','Lin. 1','Lin. 2', 'Lin.3'])
plt.xticks([i for i in range(11)], ['Conv. 1','Conv. 2','Conv. 3','Conv. 4','Conv. 5','Conv. 6','Conv. 7','Conv. 8','Lin. 1','Lin. 2', 'Lin.3'],rotation=30,)
#plt.title('Norms at')
#plt.legend()
plt.ylabel('Frobenius Norms')
plt.legend(bbox_to_anchor=(0, 1.15), loc=2)
plt.savefig('figs/norms.pdf')