import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('match_data_with_wins.csv')

#Makes wins and losses have a numerical value
df['result'] = df['result'].apply(lambda x: 1 if x == 'Win' else (0 if x == 'Loss' else x))

#Drop "Easy Columns" as they will have obvious positive and negative correlation (More goals will obviously win you more games)
columns_to_drop = ['score', 'goals', 'goals conceded', 'shots conceded', 'match_id']

#Filter out these columns, but keep 'result' for correlation calculation
df_filtered = df.drop(columns=columns_to_drop)

#Ensure that the remaining columns are numeric
df_numeric = df_filtered.select_dtypes(include=['number'])

#Compute the correlation with 'result' column separately
correlations = df_numeric.corr()

print("Correlation with Winning:")
print(correlations['result'])

#Extract and sort correlations (excluding the result column itself)
correlation_with_result = correlations['result'].drop('result')
correlation_with_result = correlation_with_result.sort_values(ascending=False)

#Horizontal Bar Plot
plt.figure(figsize=(10, 8))

colors = ['blue' if x > 0 else 'red' for x in correlation_with_result]

plt.barh(correlation_with_result.index, correlation_with_result, color=colors)
plt.xlabel('Correlation with Winning')
plt.title('Features Correlated with Winning (Positive and Negative)')
plt.axvline(x=0, color='gray', linestyle='--')
plt.show()

#Save Data and Bar Graph
correlation_with_result.to_csv('correlation_with_result.csv', header=True)
plt.savefig('correlation_with_winning.png', bbox_inches='tight') 