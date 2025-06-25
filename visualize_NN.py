import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('out/f_16/filtered_train.csv')
f = 16 # feature dimension
j = 10 # number of classes

true_labels = df['label']
print(true_labels.shape)
predicted_scores = df.iloc[:, f:f+j]  

df_melted = pd.melt(predicted_scores)
print(df_melted.shape)



# df_melted['true_label'] = true_labels.repeat(predicted_scores.shape[1]).reset_index(drop=True)
# df_melted.columns = ['Selected_Column', 'Score', 'True_Label']
# print(df_melted.shape)
# print(df_melted.loc[10000])

# g = sns.FacetGrid(df_melted, col='Selected_Column', col_wrap=5, sharex=False, sharey=False)
# g.map_dataframe(sns.histplot, x='Score', hue='True_Label')
# g.add_legend()
# g.savefig('out.png')
# g.map_dataframe(sns.histplot, x='Score', hue='True_Label', element='step', stat='density', common_norm=False)
# g.add_legend(title='True Label')

# Adjust the layout
# plt.tight_layout()
# plt.show()
