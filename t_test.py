import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('match_data_with_wins.csv')

df_numeric = df.select_dtypes(include=['number'])


columns_to_test = ['shots', 'assists', 'time on ground', 'bpm', 'saves', 'shooting percentage', 'avg boost amount', 'amount stolen', '0 boost time', 'total distance', 'time slow speed', 'time low in air', 'time high in air', 'time boost speed', 'time behind ball', 'time in front of ball', 'time defensive half', 'time offensive half', 'time defensive third', 'time neutral third', 'time offensive third', 'time ball possession', 'time ball in own side', 'demos inflicted', 'demos taken']

#Separate Wins/Losses
wins = df[df['result'] == "Win"]
losses = df[df['result'] == "Loss"]

test_results = []

#Perform T-Tests
for column in columns_to_test:
    
    
    t_stat, p_value = stats.ttest_ind(wins[column], losses[column], nan_policy='omit')
    
    
    test_results.append({
        'column': column,
        't_stat': t_stat,
        'p_value': p_value
    })

    
    print(f"T-test for {column}:")
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")
    
    
    if p_value < 0.05:
        print(f"There is a significant difference in {column} between Wins and Losses.\n")
    else:
        print(f"There is no significant difference in {column} between Wins and Losses.\n")

#Save data
test_results_df = pd.DataFrame(test_results)

#Add significance column
test_results_df['significant'] = test_results_df['p_value'] < 0.05

#Create bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='column', y='p_value', data=test_results_df, hue='significant', palette={True: 'green', False: 'red'})

plt.title("T-test Results: P-values for Each Statistic")
plt.xlabel("Statistic")
plt.ylabel("P-value")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#Save plot and Data
plt.savefig('t_test_results_plot.png')
test_results_df.to_csv('t_test_results.csv', index=False)

