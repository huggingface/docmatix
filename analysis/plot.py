import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

analysis_df = pd.read_json('prompt_analysis_results.json', orient='records', lines=True)

# Plot configurations
sns.set(style="whitegrid")
plt.figure(figsize=(16, 12))

# Plot: Number of Q/A pairs per Prompt ID
plt.subplot(3, 2, 1)
sns.barplot(x='Prompt ID', y='Number of Q/A pairs', data=analysis_df, palette='viridis')
plt.title('Number of Q/A pairs per Prompt ID')
plt.xlabel('Prompt ID')
plt.ylabel('Number of Q/A pairs')
for i, row in analysis_df.iterrows():
    plt.text(i, row['Number of Q/A pairs'], f"{row['Number of Q/A pairs']/1e6:.2f}e6", ha='center', va='bottom')

# Plot: Average answer length per Prompt ID
plt.subplot(3, 2, 2)
sns.barplot(x='Prompt ID', y='Average answer length', data=analysis_df, palette='viridis')
plt.title('Average Answer Length per Prompt ID')
plt.xlabel('Prompt ID')
plt.ylabel('Average Answer Length')
for i, row in analysis_df.iterrows():
    plt.text(i, row['Average answer length'], f"{row['Average answer length']:.2f}", ha='center', va='bottom')

# Plot: Diversity within documents per Prompt ID
plt.subplot(3, 2, 3)
sns.barplot(x='Prompt ID', y='Diversity within documents', data=analysis_df, palette='viridis')
plt.title('Diversity within Documents per Prompt ID')
plt.xlabel('Prompt ID')
plt.ylabel('Diversity within Documents')
for i, row in analysis_df.iterrows():
    plt.text(i, row['Diversity within documents'], f"{row['Diversity within documents']:.2f}", ha='center', va='bottom')

# Plot: Total empty questions per Prompt ID
plt.subplot(3, 2, 4)
sns.barplot(x='Prompt ID', y='Total empty questions', data=analysis_df, palette='viridis')
plt.title('Total Empty Questions per Prompt ID')
plt.xlabel('Prompt ID')
plt.ylabel('Total Empty Questions')
for i, row in analysis_df.iterrows():
    plt.text(i, row['Total empty questions'], f"{row['Total empty questions']}", ha='center', va='bottom')

# Plot: Average Q/A pairs per page per Prompt ID
plt.subplot(3, 2, 5)
sns.barplot(x='Prompt ID', y='Average Q/A pairs per page', data=analysis_df, palette='viridis')
plt.title('Average Q/A pairs per Page per Prompt ID')
plt.xlabel('Prompt ID')
plt.ylabel('Average Q/A pairs per Page')
for i, row in analysis_df.iterrows():
    plt.text(i, row['Average Q/A pairs per page'], f"{row['Average Q/A pairs per page']:.2f}", ha='center', va='bottom')

# Plot: Number of unique questions per Prompt ID
plt.subplot(3, 2, 6)
sns.barplot(x='Prompt ID', y='Number of unique questions', data=analysis_df, palette='viridis')
plt.title('Number of unique questions per Prompt ID')
plt.xlabel('Prompt ID')
plt.ylabel('Number of unique questions')
for i, row in analysis_df.iterrows():
    plt.text(i, row['Number of unique questions'], f"{row['Number of unique questions']/1e6:.2f}e6", ha='center', va='bottom')

plt.tight_layout()
plt.savefig('prompt_analysis_plots_enhanced.png')
plt.show()

# Summary Report
report = f"""
Prompt Analysis Report
=======================
Number of Q/A pairs per Prompt ID:
{analysis_df[['Prompt ID', 'Number of Q/A pairs']]}

Average answer length per Prompt ID:
{analysis_df[['Prompt ID', 'Average answer length']]}

Unique questions per Prompt ID:
{analysis_df[['Prompt ID', 'Number of unique questions']]}

Total pages per Prompt ID:
{analysis_df[['Prompt ID', 'Total pages']]}

Average Q/A pairs per page per Prompt ID:
{analysis_df[['Prompt ID', 'Average Q/A pairs per page']]}

Average answer length per page per Prompt ID:
{analysis_df[['Prompt ID', 'Average answer length per page']]}

Diversity within documents per Prompt ID:
{analysis_df[['Prompt ID', 'Diversity within documents']]}

Total empty questions per Prompt ID:
{analysis_df[['Prompt ID', 'Total empty questions']]}

"""

with open('prompt_analysis_report.txt', 'w') as f:
    f.write(report)

print("Report and plots generated successfully.")