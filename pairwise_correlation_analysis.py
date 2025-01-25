
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('match_data_with_wins.csv')

#Convert wins/losses to numerical
df['result'] = df['result'].apply(lambda x: 1 if x == 'Win' else (0 if x == 'Loss' else x))

#Drop "Easy Columns"
columns_to_drop = ['score', 'goals', 'goals conceded', 'shots conceded']
df_filtered = df.drop(columns=columns_to_drop)

#Ensure numerical columns
df_numeric = df_filtered.select_dtypes(include=['number'])

#Perform pairwise Pearson correlation tests
corr_matrix = pg.pairwise_corr(df_numeric)

#Display the correlation matrix with p-values
print(corr_matrix)

#Generate Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')  # Save the heatmap image
plt.tight_layout()
plt.show()

#Filter strong correlations
def filter_strong_correlations(corr_matrix, threshold=0.85):
    strong_pos_corr = []
    strong_neg_corr = []
    
    #Upper Triangle of Matrix Loop
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value >= threshold:
                strong_pos_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
            elif corr_value <= -threshold:
                strong_neg_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
    
    return strong_pos_corr, strong_neg_corr

numeric_corr_matrix = df_numeric.corr()

strong_positive, strong_negative = filter_strong_correlations(numeric_corr_matrix)

print("\nStrong Positive Correlations (>= 0.85):")
for corr in strong_positive:
    print(f"{corr[0]} and {corr[1]}: {corr[2]}")

print("\nStrong Negative Correlations (<= -0.85):")
for corr in strong_negative:
    print(f"{corr[0]} and {corr[1]}: {corr[2]}")

#Save data
corr_matrix.to_csv('pairwise_correlation_results.csv', index=False)