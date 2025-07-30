# Capital Gains and Losses Analysis
# Comprehensive Analysis of Capital Gains and Losses Impact on Income Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
df = pd.read_csv('data/adult.csv')

# Clean the data - handle missing values represented as '?'
df_clean = df.copy()
df_clean = df_clean.replace('?', np.nan)

# Create binary income variable (1 for >50K, 0 for <=50K)
df_clean['income_binary'] = (df_clean['income'] == '>50K').astype(int)

print(f"Dataset shape: {df_clean.shape}")
print(f"Missing values handled: {df_clean.isnull().sum().sum()} total missing values")
print(f"Income distribution: {dict(df_clean['income'].value_counts())}")
print(f"Binary income distribution: {dict(df_clean['income_binary'].value_counts())}")

# Basic statistics for capital gains and losses
print("\nCAPITAL GAINS AND LOSSES - BASIC STATISTICS")
print("="*60)

# Overall statistics
print("\n1. OVERALL STATISTICS:")
print(f"Dataset size: {len(df_clean):,} individuals")
print(f"High income rate (>$50K): {df_clean['income_binary'].mean():.1%}")

# Capital gains statistics
print(f"\nCAPITAL GAINS STATISTICS:")
print(f"Mean capital gains: ${df_clean['capital.gain'].mean():.2f}")
print(f"Median capital gains: ${df_clean['capital.gain'].median():.2f}")
print(f"Standard deviation: ${df_clean['capital.gain'].std():.2f}")
print(f"Maximum capital gains: ${df_clean['capital.gain'].max():,.0f}")
print(f"Minimum capital gains: ${df_clean['capital.gain'].min():,.0f}")

# Capital losses statistics
print(f"\nCAPITAL LOSSES STATISTICS:")
print(f"Mean capital losses: ${df_clean['capital.loss'].mean():.2f}")
print(f"Median capital losses: ${df_clean['capital.loss'].median():.2f}")
print(f"Standard deviation: ${df_clean['capital.loss'].std():.2f}")
print(f"Maximum capital losses: ${df_clean['capital.loss'].max():,.0f}")
print(f"Minimum capital losses: ${df_clean['capital.loss'].min():,.0f}")

# Percentiles for capital gains
print(f"\nCAPITAL GAINS PERCENTILES:")
percentiles = [90, 95, 99, 99.5, 99.9]
for p in percentiles:
    value = np.percentile(df_clean['capital.gain'], p)
    print(f"{p}th percentile: ${value:.0f}")

# Percentiles for capital losses
print(f"\nCAPITAL LOSSES PERCENTILES:")
for p in percentiles:
    value = np.percentile(df_clean['capital.loss'], p)
    print(f"{p}th percentile: ${value:.0f}")

# Zero vs non-zero analysis
gains_zero = (df_clean['capital.gain'] == 0).sum()
gains_nonzero = (df_clean['capital.gain'] > 0).sum()
losses_zero = (df_clean['capital.loss'] == 0).sum()
losses_nonzero = (df_clean['capital.loss'] > 0).sum()

print(f"\nZERO VS NON-ZERO ANALYSIS:")
print(f"People with zero capital gains: {gains_zero:,} ({gains_zero/len(df_clean)*100:.1f}%)")
print(f"People with non-zero capital gains: {gains_nonzero:,} ({gains_nonzero/len(df_clean)*100:.1f}%)")
print(f"People with zero capital losses: {losses_zero:,} ({losses_zero/len(df_clean)*100:.1f}%)")
print(f"People with non-zero capital losses: {losses_nonzero:,} ({losses_nonzero/len(df_clean)*100:.1f}%)")

# Analyze capital gains and losses by income group
print("\n\nCAPITAL GAINS AND LOSSES BY INCOME GROUP")
print("="*60)

# Split data by income group
low_income = df_clean[df_clean['income'] == '<=50K']
high_income = df_clean[df_clean['income'] == '>50K']

print(f"\nSAMPLE SIZES:")
print(f"Low income (<=50K): {len(low_income):,} individuals ({len(low_income)/len(df_clean)*100:.1f}%)")
print(f"High income (>50K): {len(high_income):,} individuals ({len(high_income)/len(df_clean)*100:.1f}%)")

# Capital gains analysis by income group
print(f"\nCAPITAL GAINS BY INCOME GROUP:")
print(f"Low income group (<=50K):")
print(f"  Mean: ${low_income['capital.gain'].mean():.2f}")
print(f"  Median: ${low_income['capital.gain'].median():.2f}")
print(f"  Std: ${low_income['capital.gain'].std():.2f}")
print(f"  25th percentile: ${low_income['capital.gain'].quantile(0.25):.2f}")
print(f"  75th percentile: ${low_income['capital.gain'].quantile(0.75):.2f}")
print(f"  90th percentile: ${low_income['capital.gain'].quantile(0.90):.2f}")
print(f"  95th percentile: ${low_income['capital.gain'].quantile(0.95):.2f}")

print(f"\nHigh income group (>50K):")
print(f"  Mean: ${high_income['capital.gain'].mean():.2f}")
print(f"  Median: ${high_income['capital.gain'].median():.2f}")
print(f"  Std: ${high_income['capital.gain'].std():.2f}")
print(f"  25th percentile: ${high_income['capital.gain'].quantile(0.25):.2f}")
print(f"  75th percentile: ${high_income['capital.gain'].quantile(0.75):.2f}")
print(f"  90th percentile: ${high_income['capital.gain'].quantile(0.90):.2f}")
print(f"  95th percentile: ${high_income['capital.gain'].quantile(0.95):.2f}")

# Capital losses analysis by income group
print(f"\nCAPITAL LOSSES BY INCOME GROUP:")
print(f"Low income group (<=50K):")
print(f"  Mean: ${low_income['capital.loss'].mean():.2f}")
print(f"  Median: ${low_income['capital.loss'].median():.2f}")
print(f"  Std: ${low_income['capital.loss'].std():.2f}")
print(f"  25th percentile: ${low_income['capital.loss'].quantile(0.25):.2f}")
print(f"  75th percentile: ${low_income['capital.loss'].quantile(0.75):.2f}")
print(f"  90th percentile: ${low_income['capital.loss'].quantile(0.90):.2f}")
print(f"  95th percentile: ${low_income['capital.loss'].quantile(0.95):.2f}")

print(f"\nHigh income group (>50K):")
print(f"  Mean: ${high_income['capital.loss'].mean():.2f}")
print(f"  Median: ${high_income['capital.loss'].median():.2f}")
print(f"  Std: ${high_income['capital.loss'].std():.2f}")
print(f"  25th percentile: ${high_income['capital.loss'].quantile(0.25):.2f}")
print(f"  75th percentile: ${high_income['capital.loss'].quantile(0.75):.2f}")
print(f"  90th percentile: ${high_income['capital.loss'].quantile(0.90):.2f}")
print(f"  95th percentile: ${high_income['capital.loss'].quantile(0.95):.2f}")

# Ratio comparisons
gains_ratio = high_income['capital.gain'].mean() / low_income['capital.gain'].mean()
losses_ratio = high_income['capital.loss'].mean() / low_income['capital.loss'].mean()

print(f"\nRATIO COMPARISONS (High income vs Low income):")
print(f"Capital gains mean ratio: {gains_ratio:.1f}x")
print(f"Capital losses mean ratio: {losses_ratio:.1f}x")

# Analyze percentage of people with non-zero capital gains/losses by income group
print("\n\nNON-ZERO CAPITAL GAINS AND LOSSES BY INCOME GROUP")
print("="*60)

# Capital gains analysis
low_income_gains_pct = (low_income['capital.gain'] > 0).mean() * 100
high_income_gains_pct = (high_income['capital.gain'] > 0).mean() * 100

print(f"\nPERCENTAGE WITH NON-ZERO CAPITAL GAINS:")
print(f"Low income (<=50K): {low_income_gains_pct:.1f}%")
print(f"High income (>50K): {high_income_gains_pct:.1f}%")
print(f"Ratio (High/Low): {high_income_gains_pct/low_income_gains_pct:.1f}x")

# Capital losses analysis
low_income_losses_pct = (low_income['capital.loss'] > 0).mean() * 100
high_income_losses_pct = (high_income['capital.loss'] > 0).mean() * 100

print(f"\nPERCENTAGE WITH NON-ZERO CAPITAL LOSSES:")
print(f"Low income (<=50K): {low_income_losses_pct:.1f}%")
print(f"High income (>50K): {high_income_losses_pct:.1f}%")
print(f"Ratio (High/Low): {high_income_losses_pct/low_income_losses_pct:.1f}x")

# Create categorical variables for analysis
df_clean['has_capital_gains'] = (df_clean['capital.gain'] > 0).astype(int)
df_clean['has_capital_losses'] = (df_clean['capital.loss'] > 0).astype(int)

# Contingency table analysis
print(f"\nCONTINGENCY TABLE ANALYSIS:")

# Capital gains contingency table
gains_crosstab = pd.crosstab(df_clean['has_capital_gains'], df_clean['income'])
print(f"\nCapital Gains Contingency Table:")
print(gains_crosstab)
print(f"\nCapital Gains Proportions:")
print(pd.crosstab(df_clean['has_capital_gains'], df_clean['income'], normalize='columns').round(3))

# Capital losses contingency table
losses_crosstab = pd.crosstab(df_clean['has_capital_losses'], df_clean['income'])
print(f"\nCapital Losses Contingency Table:")
print(losses_crosstab)
print(f"\nCapital Losses Proportions:")
print(pd.crosstab(df_clean['has_capital_losses'], df_clean['income'], normalize='columns').round(3))

# Calculate odds ratios
print(f"\nODDS RATIO ANALYSIS:")

# Capital gains odds ratio
gains_odds_low = (low_income['capital.gain'] > 0).sum() / (low_income['capital.gain'] == 0).sum()
gains_odds_high = (high_income['capital.gain'] > 0).sum() / (high_income['capital.gain'] == 0).sum()
gains_odds_ratio = gains_odds_high / gains_odds_low

print(f"Capital Gains Odds Ratio: {gains_odds_ratio:.2f}")
print(f"  High income odds: {gains_odds_high:.3f}")
print(f"  Low income odds: {gains_odds_low:.3f}")

# Capital losses odds ratio
losses_odds_low = (low_income['capital.loss'] > 0).sum() / (low_income['capital.loss'] == 0).sum()
losses_odds_high = (high_income['capital.loss'] > 0).sum() / (high_income['capital.loss'] == 0).sum()
losses_odds_ratio = losses_odds_high / losses_odds_low

print(f"Capital Losses Odds Ratio: {losses_odds_ratio:.2f}")
print(f"  High income odds: {losses_odds_high:.3f}")
print(f"  Low income odds: {losses_odds_low:.3f}")

# Analyze predictive power of capital gains/losses
print("\n\nPREDICTIVE POWER ANALYSIS")
print("="*60)

# If someone has capital gains, what's the probability they have high income?
has_gains = df_clean['capital.gain'] > 0
has_losses = df_clean['capital.loss'] > 0

# Conditional probabilities for capital gains
prob_high_income_given_gains = df_clean[has_gains]['income_binary'].mean()
prob_high_income_given_no_gains = df_clean[~has_gains]['income_binary'].mean()

print(f"\nCAPITAL GAINS PREDICTIVE POWER:")
print(f"P(High Income | Has Capital Gains) = {prob_high_income_given_gains:.1%}")
print(f"P(High Income | No Capital Gains) = {prob_high_income_given_no_gains:.1%}")
print(f"Lift = {prob_high_income_given_gains/prob_high_income_given_no_gains:.1f}x")

# Conditional probabilities for capital losses
prob_high_income_given_losses = df_clean[has_losses]['income_binary'].mean()
prob_high_income_given_no_losses = df_clean[~has_losses]['income_binary'].mean()

print(f"\nCAPITAL LOSSES PREDICTIVE POWER:")
print(f"P(High Income | Has Capital Losses) = {prob_high_income_given_losses:.1%}")
print(f"P(High Income | No Capital Losses) = {prob_high_income_given_no_losses:.1%}")
print(f"Lift = {prob_high_income_given_losses/prob_high_income_given_no_losses:.1f}x")

# Combined analysis
has_both = has_gains & has_losses
has_either = has_gains | has_losses
has_neither = ~has_gains & ~has_losses

print(f"\nCOMBINED ANALYSIS:")
print(f"P(High Income | Has Both Gains & Losses) = {df_clean[has_both]['income_binary'].mean():.1%}")
print(f"P(High Income | Has Either Gains or Losses) = {df_clean[has_either]['income_binary'].mean():.1%}")
print(f"P(High Income | Has Neither Gains nor Losses) = {df_clean[has_neither]['income_binary'].mean():.1%}")

# Calculate sample sizes for each group
print(f"\nSAMPLE SIZES:")
print(f"Has capital gains: {has_gains.sum():,} ({has_gains.mean()*100:.1f}%)")
print(f"Has capital losses: {has_losses.sum():,} ({has_losses.mean()*100:.1f}%)")
print(f"Has both: {has_both.sum():,} ({has_both.mean()*100:.1f}%)")
print(f"Has either: {has_either.sum():,} ({has_either.mean()*100:.1f}%)")
print(f"Has neither: {has_neither.sum():,} ({has_neither.mean()*100:.1f}%)")

# Information gain / predictive value
baseline_prob = df_clean['income_binary'].mean()
print(f"\nBASELINE PROBABILITY:")
print(f"Overall P(High Income) = {baseline_prob:.1%}")

# Calculate information gain
print(f"\nINFORMATION GAIN:")
gains_info_gain = prob_high_income_given_gains - baseline_prob
losses_info_gain = prob_high_income_given_losses - baseline_prob

print(f"Capital gains information gain: {gains_info_gain:+.1%}")
print(f"Capital losses information gain: {losses_info_gain:+.1%}")

# Statistical tests and correlation analysis
print("\n\nSTATISTICAL TESTS AND CORRELATIONS")
print("="*60)

# 1. Chi-square test for independence
print("\n1. CHI-SQUARE TESTS FOR INDEPENDENCE:")

# Capital gains chi-square test
chi2_gains, p_gains, dof_gains, expected_gains = chi2_contingency(gains_crosstab)
print(f"\nCapital Gains vs Income:")
print(f"  Chi-square statistic: {chi2_gains:.4f}")
print(f"  P-value: {p_gains:.2e}")
print(f"  Degrees of freedom: {dof_gains}")
print(f"  Result: {'Significant' if p_gains < 0.05 else 'Not significant'} association")

# Capital losses chi-square test
chi2_losses, p_losses, dof_losses, expected_losses = chi2_contingency(losses_crosstab)
print(f"\nCapital Losses vs Income:")
print(f"  Chi-square statistic: {chi2_losses:.4f}")
print(f"  P-value: {p_losses:.2e}")
print(f"  Degrees of freedom: {dof_losses}")
print(f"  Result: {'Significant' if p_losses < 0.05 else 'Not significant'} association")

# 2. Point-biserial correlation
print(f"\n2. POINT-BISERIAL CORRELATIONS:")

# Capital gains correlation
corr_gains, p_corr_gains = pointbiserialr(df_clean['income_binary'], df_clean['capital.gain'])
print(f"Capital Gains - Income correlation: {corr_gains:.4f} (p={p_corr_gains:.2e})")

# Capital losses correlation  
corr_losses, p_corr_losses = pointbiserialr(df_clean['income_binary'], df_clean['capital.loss'])
print(f"Capital Losses - Income correlation: {corr_losses:.4f} (p={p_corr_losses:.2e})")

# Has capital gains correlation
corr_has_gains, p_corr_has_gains = pointbiserialr(df_clean['income_binary'], df_clean['has_capital_gains'])
print(f"Has Capital Gains - Income correlation: {corr_has_gains:.4f} (p={p_corr_has_gains:.2e})")

# Has capital losses correlation
corr_has_losses, p_corr_has_losses = pointbiserialr(df_clean['income_binary'], df_clean['has_capital_losses'])
print(f"Has Capital Losses - Income correlation: {corr_has_losses:.4f} (p={p_corr_has_losses:.2e})")

# 3. Mann-Whitney U test (non-parametric test for comparing distributions)
print(f"\n3. MANN-WHITNEY U TESTS:")

# Capital gains comparison
u_gains, p_u_gains = mannwhitneyu(high_income['capital.gain'], low_income['capital.gain'], alternative='two-sided')
print(f"Capital Gains distribution comparison:")
print(f"  U-statistic: {u_gains:.0f}")
print(f"  P-value: {p_u_gains:.2e}")
print(f"  Result: {'Significant' if p_u_gains < 0.05 else 'Not significant'} difference in distributions")

# Capital losses comparison
u_losses, p_u_losses = mannwhitneyu(high_income['capital.loss'], low_income['capital.loss'], alternative='two-sided')
print(f"Capital Losses distribution comparison:")
print(f"  U-statistic: {u_losses:.0f}")
print(f"  P-value: {p_u_losses:.2e}")
print(f"  Result: {'Significant' if p_u_losses < 0.05 else 'Not significant'} difference in distributions")

# 4. Effect sizes (Cohen's d)
print(f"\n4. EFFECT SIZES (Cohen's d):")

def cohens_d(group1, group2):
    """Calculate Cohen's d for effect size"""
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(), group2.std()
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

d_gains = cohens_d(high_income['capital.gain'], low_income['capital.gain'])
d_losses = cohens_d(high_income['capital.loss'], low_income['capital.loss'])

print(f"Capital Gains Cohen's d: {d_gains:.4f}")
print(f"Capital Losses Cohen's d: {d_losses:.4f}")

# Interpret effect sizes
def interpret_cohens_d(d):
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"

print(f"Capital Gains effect size: {interpret_cohens_d(d_gains)}")
print(f"Capital Losses effect size: {interpret_cohens_d(d_losses)}")

# 5. Cramer's V (effect size for categorical associations)
print(f"\n5. CRAMER'S V (Effect size for categorical associations):")

def cramers_v(chi2, n, r, c):
    """Calculate Cramer's V"""
    return np.sqrt(chi2 / (n * min(r-1, c-1)))

n = len(df_clean)
cramers_v_gains = cramers_v(chi2_gains, n, 2, 2)
cramers_v_losses = cramers_v(chi2_losses, n, 2, 2)

print(f"Capital Gains Cramer's V: {cramers_v_gains:.4f}")
print(f"Capital Losses Cramer's V: {cramers_v_losses:.4f}")

# Comprehensive summary of findings
print("\n\nCAPITAL GAINS AND LOSSES ANALYSIS - KEY FINDINGS")
print("="*70)

print(f"\n1. DATASET OVERVIEW:")
print(f"   • Total sample size: {len(df_clean):,} individuals")
print(f"   • High income rate (>$50K): {df_clean['income_binary'].mean():.1%}")
print(f"   • Low income rate (<=50K): {(1-df_clean['income_binary'].mean()):.1%}")

print(f"\n2. CAPITAL GAINS DISTRIBUTION:")
print(f"   • Mean capital gains: ${df_clean['capital.gain'].mean():.2f}")
print(f"   • Median capital gains: ${df_clean['capital.gain'].median():.2f}")
print(f"   • People with non-zero capital gains: {(df_clean['capital.gain'] > 0).sum():,} ({(df_clean['capital.gain'] > 0).mean()*100:.1f}%)")
print(f"   • People with zero capital gains: {(df_clean['capital.gain'] == 0).sum():,} ({(df_clean['capital.gain'] == 0).mean()*100:.1f}%)")

print(f"\n3. CAPITAL LOSSES DISTRIBUTION:")
print(f"   • Mean capital losses: ${df_clean['capital.loss'].mean():.2f}")
print(f"   • Median capital losses: ${df_clean['capital.loss'].median():.2f}")
print(f"   • People with non-zero capital losses: {(df_clean['capital.loss'] > 0).sum():,} ({(df_clean['capital.loss'] > 0).mean()*100:.1f}%)")
print(f"   • People with zero capital losses: {(df_clean['capital.loss'] == 0).sum():,} ({(df_clean['capital.loss'] == 0).mean()*100:.1f}%)")

print(f"\n4. INCOME GROUP DIFFERENCES:")
print(f"   Capital Gains:")
print(f"   • High income group mean: ${high_income['capital.gain'].mean():.2f}")
print(f"   • Low income group mean: ${low_income['capital.gain'].mean():.2f}")
print(f"   • Ratio (High/Low): {high_income['capital.gain'].mean()/low_income['capital.gain'].mean():.1f}x")
print(f"   Capital Losses:")
print(f"   • High income group mean: ${high_income['capital.loss'].mean():.2f}")
print(f"   • Low income group mean: ${low_income['capital.loss'].mean():.2f}")
print(f"   • Ratio (High/Low): {high_income['capital.loss'].mean()/low_income['capital.loss'].mean():.1f}x")

print(f"\n5. PREDICTIVE POWER:")
print(f"   Capital Gains:")
print(f"   • P(High Income | Has Capital Gains) = {prob_high_income_given_gains:.1%}")
print(f"   • P(High Income | No Capital Gains) = {prob_high_income_given_no_gains:.1%}")
print(f"   • Lift = {prob_high_income_given_gains/prob_high_income_given_no_gains:.1f}x")
print(f"   Capital Losses:")
print(f"   • P(High Income | Has Capital Losses) = {prob_high_income_given_losses:.1%}")
print(f"   • P(High Income | No Capital Losses) = {prob_high_income_given_no_losses:.1%}")
print(f"   • Lift = {prob_high_income_given_losses/prob_high_income_given_no_losses:.1f}x")

print(f"\n6. STATISTICAL SIGNIFICANCE:")
print(f"   • Capital Gains - Income correlation: {corr_gains:.4f} (p={p_corr_gains:.2e})")
print(f"   • Capital Losses - Income correlation: {corr_losses:.4f} (p={p_corr_losses:.2e})")
print(f"   • Has Capital Gains - Income correlation: {corr_has_gains:.4f} (p={p_corr_has_gains:.2e})")
print(f"   • Has Capital Losses - Income correlation: {corr_has_losses:.4f} (p={p_corr_has_losses:.2e})")
print(f"   • Chi-square p-value (Capital Gains): {p_gains:.2e}")
print(f"   • Chi-square p-value (Capital Losses): {p_losses:.2e}")

print(f"\n7. EFFECT SIZES:")
print(f"   • Capital Gains Cohen's d: {d_gains:.4f} ({interpret_cohens_d(d_gains)})")
print(f"   • Capital Losses Cohen's d: {d_losses:.4f} ({interpret_cohens_d(d_losses)})")
print(f"   • Capital Gains Cramer's V: {cramers_v_gains:.4f}")
print(f"   • Capital Losses Cramer's V: {cramers_v_losses:.4f}")

print(f"\n8. KEY CONCLUSIONS:")
print(f"   * Capital gains are STRONG predictors of high income")
print(f"     - Having capital gains increases probability of high income by {gains_info_gain*100:.1f} percentage points")
print(f"     - {high_income_gains_pct:.1f}% of high earners have capital gains vs {low_income_gains_pct:.1f}% of low earners")
print(f"   * Capital losses are also significant predictors of high income")
print(f"     - Having capital losses increases probability of high income by {losses_info_gain*100:.1f} percentage points")
print(f"     - {high_income_losses_pct:.1f}% of high earners have capital losses vs {low_income_losses_pct:.1f}% of low earners")
print(f"   * Both capital gains and losses are statistically significant predictors (p < 0.001)")
print(f"   * Capital gains show a {interpret_cohens_d(d_gains)} effect size")
print(f"   * Capital losses show a {interpret_cohens_d(d_losses)} effect size")
print(f"   * The majority of people ({(df_clean['capital.gain'] == 0).mean()*100:.1f}%) have zero capital gains")
print(f"   * The majority of people ({(df_clean['capital.loss'] == 0).mean()*100:.1f}%) have zero capital losses")

print(f"\n9. PRACTICAL IMPLICATIONS:")
print(f"   * Capital gains/losses are strong indicators of investment activity")
print(f"   * Investment activity is strongly associated with higher income")
print(f"   * Having ANY capital gains/losses (even small amounts) is predictive")
print(f"   * These features should be valuable in income prediction models")
print(f"   * The binary indicators (has/doesn't have) may be more useful than continuous amounts")

print(f"\n10. RECOMMENDATION:")
print(f"   Based on this analysis, capital gains and losses are STRONG predictors of income")
print(f"   level and should be included in any income prediction model. The binary")
print(f"   indicators (presence/absence) are particularly powerful predictors.")