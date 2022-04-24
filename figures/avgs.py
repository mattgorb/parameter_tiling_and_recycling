import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from seaborn import lineplot,scatterplot
df1 = pd.read_csv(f'tables/model_combos_conv6-sc-epu-var_0.5.csv')
print(np.mean(df1['percent_same_total'].values))
print(np.mean(df1['jaccard_total'].values))

print(np.std(df1['percent_same_total'].values))
print(np.std(df1['jaccard_total'].values))

df1 = pd.read_csv(f'tables/model_combos_conv6-sc-epu-var_0.25.csv')
print(np.mean(df1['percent_same_total'].values))
print(np.mean(df1['jaccard_total'].values))

df1 = pd.read_csv(f'tables/model_combos_conv6-kn-biprop-var_0.5.csv')
print(np.mean(df1['percent_same_total'].values))
print(np.mean(df1['jaccard_total'].values))

df1 = pd.read_csv(f'tables/model_combos_conv6-kn-biprop-var_0.25.csv')
print(np.mean(df1['percent_same_total'].values))
print(np.mean(df1['jaccard_total'].values))