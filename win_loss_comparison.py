import pandas as pd
import matplotlib.pyplot as plt
import data
import seaborn as sns

df = data.df

#Group matches
df['match_id'] = df.index // 2

#Determine win/loss
df['result'] = df.groupby('match_id')['score'].transform(lambda x: x == x.max())
df['result'] = df['result'].replace({True: 'Win', False: 'Loss'})

#Count wins
win_counts = df[df['result'] == 'Win']['team name'].value_counts()
print("Win counts per team:")
print(win_counts)

matches = []
for match_id, match_data in df.groupby('match_id'):
    winner = match_data[match_data['result'] == 'Win']
    loser = match_data[match_data['result'] == 'Loss']
    
    if winner.empty or loser.empty:
        continue  
    
    winner_team = winner['team name'].values[0]
    loser_team = loser['team name'].values[0]
    
    #Select numeric columns
    numeric_columns = match_data.select_dtypes(include=['number']).columns
    
    #Corece numerics
    match_data[numeric_columns] = match_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    #Check NaN Values
    if match_data[numeric_columns].isna().sum().sum() > 0:
        print(f"Warning: NaN values detected in match {match_id}, filling with zeros.")
        match_data[numeric_columns] = match_data[numeric_columns].fillna(0)
    
    stat_differences = winner[numeric_columns].iloc[0] - loser[numeric_columns].iloc[0]
    
    
    matches.append({
        'winner': winner_team,
        'loser': loser_team,
        'stat_differences': stat_differences
    })

#Display results
for match in matches:
    print(f"\nWinner: {match['winner']}")
    print(f"Loser: {match['loser']}")
    for stat, diff in match['stat_differences'].items():
        
        if stat != 'match_id':
            print(f"{stat}: {diff:+.2f}")

df.to_csv('match_data_with_wins.csv', index=False)

stat_differences_list = []


for match in matches:
    for stat, diff in match['stat_differences'].items():
        if stat != 'match_id':
            stat_differences_list.append({
                'match_id': match['winner'] + " vs " + match['loser'],
                'stat': stat,
                'difference': diff
            })

#Convert the list of dictionaries to a DataFrame
stat_differences_df = pd.DataFrame(stat_differences_list)

#Take out total distance as its a Huge number compared to other stats
stat_differences_filtered = stat_differences_df[stat_differences_df['stat'] != 'total distance']

#Plot Differences
plt.figure(figsize=(12, 6))
sns.barplot(x='stat', y='difference', data=stat_differences_filtered, hue='match_id')
plt.xticks(rotation=90)
plt.title('Stat Differences Between Winners and Losers')
plt.tight_layout()  # Adjust layout to fit labels
plt.show()


#Save data and chart
stat_differences_filtered.to_csv('filtered_stat_differences.csv', index=False)
plt.savefig('stat_differences_plot.png')