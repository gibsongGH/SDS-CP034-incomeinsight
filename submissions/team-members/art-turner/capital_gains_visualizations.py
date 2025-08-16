import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('data/adult.csv')
df_clean = df.copy()
df_clean = df_clean.replace('?', np.nan)
df_clean['income_binary'] = (df_clean['income'] == '>50K').astype(int)

# Split data by income group
low_income = df_clean[df_clean['income'] == '<=50K']
high_income = df_clean[df_clean['income'] == '>50K']

# Calculate percentages
low_income_gains_pct = (low_income['capital.gain'] > 0).mean() * 100
high_income_gains_pct = (high_income['capital.gain'] > 0).mean() * 100
low_income_losses_pct = (low_income['capital.loss'] > 0).mean() * 100
high_income_losses_pct = (high_income['capital.loss'] > 0).mean() * 100

# Calculate probabilities
has_gains = df_clean['capital.gain'] > 0
has_losses = df_clean['capital.loss'] > 0
prob_high_income_given_gains = df_clean[has_gains]['income_binary'].mean()
prob_high_income_given_no_gains = df_clean[~has_gains]['income_binary'].mean()
prob_high_income_given_losses = df_clean[has_losses]['income_binary'].mean()
prob_high_income_given_no_losses = df_clean[~has_losses]['income_binary'].mean()
baseline_prob = df_clean['income_binary'].mean()

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Capital Gains and Losses Analysis - Income Impact', fontsize=16, fontweight='bold')

# 1. Distribution of capital gains by income group (log scale for better visibility)
axes[0, 0].hist(low_income['capital.gain'][low_income['capital.gain'] > 0], 
                bins=50, alpha=0.7, label='<=50K', color='lightcoral', density=True)
axes[0, 0].hist(high_income['capital.gain'][high_income['capital.gain'] > 0], 
                bins=50, alpha=0.7, label='>50K', color='lightblue', density=True)
axes[0, 0].set_title('Non-Zero Capital Gains Distribution\nby Income Group')
axes[0, 0].set_xlabel('Capital Gains ($)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()
axes[0, 0].set_xscale('log')

# 2. Distribution of capital losses by income group
axes[0, 1].hist(low_income['capital.loss'][low_income['capital.loss'] > 0], 
                bins=30, alpha=0.7, label='<=50K', color='lightcoral', density=True)
axes[0, 1].hist(high_income['capital.loss'][high_income['capital.loss'] > 0], 
                bins=30, alpha=0.7, label='>50K', color='lightblue', density=True)
axes[0, 1].set_title('Non-Zero Capital Losses Distribution\nby Income Group')
axes[0, 1].set_xlabel('Capital Losses ($)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

# 3. Box plots comparison (limited range for visibility)
gains_data = [low_income['capital.gain'], high_income['capital.gain']]
bp1 = axes[0, 2].boxplot(gains_data, labels=['<=50K', '>50K'], patch_artist=True)
bp1['boxes'][0].set_facecolor('lightcoral')
bp1['boxes'][1].set_facecolor('lightblue')
axes[0, 2].set_title('Capital Gains Distribution\nby Income Group')
axes[0, 2].set_ylabel('Capital Gains ($)')
axes[0, 2].set_ylim(0, 20000)  # Limit for better visibility

# 4. Percentage with non-zero capital gains/losses
categories = ['Has Capital\nGains', 'Has Capital\nLosses']
low_income_pcts = [low_income_gains_pct, low_income_losses_pct]
high_income_pcts = [high_income_gains_pct, high_income_losses_pct]

x = np.arange(len(categories))
width = 0.35

bars1 = axes[1, 0].bar(x - width/2, low_income_pcts, width, label='<=50K', color='lightcoral')
bars2 = axes[1, 0].bar(x + width/2, high_income_pcts, width, label='>50K', color='lightblue')
axes[1, 0].set_title('Percentage with Non-Zero\nCapital Gains/Losses')
axes[1, 0].set_ylabel('Percentage (%)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(categories)
axes[1, 0].legend()

# Add value labels on bars
for i, (low, high) in enumerate(zip(low_income_pcts, high_income_pcts)):
    axes[1, 0].text(i - width/2, low + 0.5, f'{low:.1f}%', ha='center', va='bottom')
    axes[1, 0].text(i + width/2, high + 0.5, f'{high:.1f}%', ha='center', va='bottom')

# 5. Predictive power visualization
groups = ['Has\nGains', 'No\nGains', 'Has\nLosses', 'No\nLosses']
probabilities = [prob_high_income_given_gains, prob_high_income_given_no_gains, 
                prob_high_income_given_losses, prob_high_income_given_no_losses]
colors = ['darkgreen', 'lightgreen', 'darkred', 'lightpink']

bars = axes[1, 1].bar(groups, [p*100 for p in probabilities], color=colors)
axes[1, 1].set_title('High Income Probability\nby Capital Gains/Losses')
axes[1, 1].set_ylabel('Probability of High Income (%)')

# Add baseline line
axes[1, 1].axhline(y=baseline_prob*100, color='black', linestyle='--', 
                   alpha=0.7, label=f'Baseline ({baseline_prob*100:.1f}%)')
axes[1, 1].legend()

# Add value labels on bars
for bar, prob in zip(bars, probabilities):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prob*100:.1f}%', ha='center', va='bottom')

# 6. Mean capital gains and losses by income group
income_groups = ['<=50K', '>50K']
means_gains = [low_income['capital.gain'].mean(), high_income['capital.gain'].mean()]
means_losses = [low_income['capital.loss'].mean(), high_income['capital.loss'].mean()]

x = np.arange(len(income_groups))
width = 0.35

bars1 = axes[1, 2].bar(x - width/2, means_gains, width, label='Capital Gains', color='green', alpha=0.7)
bars2 = axes[1, 2].bar(x + width/2, means_losses, width, label='Capital Losses', color='red', alpha=0.7)
axes[1, 2].set_title('Mean Capital Gains/Losses\nby Income Group')
axes[1, 2].set_ylabel('Mean Amount ($)')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(income_groups)
axes[1, 2].legend()

# Add value labels on bars
for i, (gains, losses) in enumerate(zip(means_gains, means_losses)):
    axes[1, 2].text(i - width/2, gains + 50, f'${gains:.0f}', ha='center', va='bottom')
    axes[1, 2].text(i + width/2, losses + 50, f'${losses:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('capital_gains_analysis_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations created and saved as 'capital_gains_analysis_visualizations.png'")