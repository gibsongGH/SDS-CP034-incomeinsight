#!/usr/bin/env python3
"""
Comprehensive Income Disparity Analysis
Analysis of income disparities across race, sex, and native country in the Adult dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, proportions_ztest
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette('husl')

# Load the dataset
print("Loading Adult dataset...")
df = pd.read_csv('data/adult.csv')
print(f"Dataset shape: {df.shape}")

# Data cleaning for disparity analysis
print("\nCleaning data...")
df_clean = df.copy()
df_clean = df_clean.replace('?', np.nan)

# Remove rows with missing values in key demographic variables
key_vars = ['race', 'sex', 'native.country', 'income']
df_clean = df_clean.dropna(subset=key_vars)

print(f"Original dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {df_clean.shape}")
print(f"Removed {df.shape[0] - df_clean.shape[0]} rows with missing demographic data")

# Create binary income variable
df_clean['high_income'] = (df_clean['income'] == '>50K').astype(int)

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS OF INCOME DISPARITIES")
print("="*80)

# Overall statistics
total_sample = len(df_clean)
overall_high_income_rate = df_clean['high_income'].mean()

print(f"\nOVERALL STATISTICS:")
print(f"Total sample size: {total_sample:,}")
print(f"Overall high income rate (>$50K): {overall_high_income_rate:.1%}")

# 1. RACE ANALYSIS
print(f"\n1. RACE DISPARITIES:")
print("-" * 40)
race_income = pd.crosstab(df_clean['race'], df_clean['income'], normalize='index')
race_high_income = race_income['>50K'].sort_values(ascending=False)

race_high_income_summary = df_clean.groupby('race')['high_income'].agg(['count', 'mean']).round(4)
race_high_income_summary.columns = ['Sample_Size', 'High_Income_Rate']
race_high_income_summary = race_high_income_summary.sort_values('High_Income_Rate', ascending=False)

for race, row in race_high_income_summary.iterrows():
    deviation = row['High_Income_Rate'] - overall_high_income_rate
    print(f"{race}: {row['High_Income_Rate']:.1%} ({deviation:+.1%} from average, n={row['Sample_Size']:,})")

# 2. SEX ANALYSIS
print(f"\n2. SEX DISPARITIES:")
print("-" * 40)
sex_income = pd.crosstab(df_clean['sex'], df_clean['income'], normalize='index')
sex_high_income = sex_income['>50K'].sort_values(ascending=False)

sex_high_income_summary = df_clean.groupby('sex')['high_income'].agg(['count', 'mean']).round(4)
sex_high_income_summary.columns = ['Sample_Size', 'High_Income_Rate']
sex_high_income_summary = sex_high_income_summary.sort_values('High_Income_Rate', ascending=False)

for sex, row in sex_high_income_summary.iterrows():
    deviation = row['High_Income_Rate'] - overall_high_income_rate
    print(f"{sex}: {row['High_Income_Rate']:.1%} ({deviation:+.1%} from average, n={row['Sample_Size']:,})")

# Calculate the gender pay gap
male_high_income = sex_high_income['Male']
female_high_income = sex_high_income['Female']
gender_gap = male_high_income - female_high_income

print(f"\nGender Income Gap Analysis:")
print(f"Male high income rate: {male_high_income:.1%}")
print(f"Female high income rate: {female_high_income:.1%}")
print(f"Gender gap: {gender_gap:.1f} percentage points")
print(f"Males are {male_high_income/female_high_income:.1f}x more likely to earn >$50K")

# 3. NATIVE COUNTRY ANALYSIS
print(f"\n3. NATIVE COUNTRY DISPARITIES:")
print("-" * 40)

# Focus on countries with meaningful sample sizes (>50 individuals)
country_counts = df_clean['native.country'].value_counts()
min_sample_size = 50
significant_countries = country_counts[country_counts >= min_sample_size].index.tolist()

print(f"Countries with ≥{min_sample_size} individuals: {len(significant_countries)}")
print("Top 10 countries by population:")
for i, (country, count) in enumerate(country_counts.head(10).items(), 1):
    print(f"{i:2d}. {country}: {count:,}")

# Filter dataset for significant countries
df_countries = df_clean[df_clean['native.country'].isin(significant_countries)].copy()

country_income = pd.crosstab(df_countries['native.country'], df_countries['income'], normalize='index')
country_high_income = country_income['>50K'].sort_values(ascending=False)

country_high_income_summary = df_countries.groupby('native.country')['high_income'].agg(['count', 'mean']).round(4)
country_high_income_summary.columns = ['Sample_Size', 'High_Income_Rate']
country_high_income_summary = country_high_income_summary.sort_values('High_Income_Rate', ascending=False)

print(f"\nHigh income rates by native country (countries with ≥{min_sample_size} individuals):")
for country, row in country_high_income_summary.iterrows():
    deviation = row['High_Income_Rate'] - overall_high_income_rate
    print(f"{country}: {row['High_Income_Rate']:.1%} ({deviation:+.1%} from average, n={row['Sample_Size']:,})")

# 4. INTERSECTIONAL ANALYSIS
print(f"\n4. INTERSECTIONAL ANALYSIS (Race and Sex):")
print("-" * 40)

race_sex_income = pd.crosstab([df_clean['race'], df_clean['sex']], df_clean['income'], normalize='index')
race_sex_high_income = race_sex_income['>50K'].sort_values(ascending=False)

print(f"High income rates by race and sex:")
for (race, sex), pct in race_sex_high_income.items():
    sample_size = df_clean[(df_clean['race'] == race) & (df_clean['sex'] == sex)].shape[0]
    print(f"{race} {sex}: {pct:.1%} (n={sample_size:,})")

# Calculate gender gaps within each race
race_sex_pivot = race_sex_income['>50K'].unstack()
print(f"\nGender gaps within each race:")
for race in race_sex_pivot.index:
    male_rate = race_sex_pivot.loc[race, 'Male']
    female_rate = race_sex_pivot.loc[race, 'Female']
    gap = male_rate - female_rate
    ratio = male_rate / female_rate if female_rate > 0 else float('inf')
    print(f"{race}: {gap:.1f} percentage points gap, {ratio:.1f}x ratio")

# 5. STATISTICAL SIGNIFICANCE TESTING
print(f"\n5. STATISTICAL SIGNIFICANCE TESTING:")
print("-" * 40)

# Test 1: Chi-square test for race and income
race_income_table = pd.crosstab(df_clean['race'], df_clean['income'])
chi2, p_value, dof, expected = chi2_contingency(race_income_table)
print(f"Race and income association:")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  P-value: {p_value:.2e}")
print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} association")

# Test 2: Chi-square test for sex and income
sex_income_table = pd.crosstab(df_clean['sex'], df_clean['income'])
chi2, p_value, dof, expected = chi2_contingency(sex_income_table)
print(f"\nSex and income association:")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  P-value: {p_value:.2e}")
print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} association")

# Test 3: Chi-square test for native country and income
country_income_table = pd.crosstab(df_countries['native.country'], df_countries['income'])
chi2, p_value, dof, expected = chi2_contingency(country_income_table)
print(f"\nNative country and income association:")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  P-value: {p_value:.2e}")
print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} association")

# Test 4: Pairwise comparisons for race (vs White as reference)
print(f"\nPairwise race comparisons (vs White as reference):")
white_high = df_clean[df_clean['race'] == 'White']['high_income']
white_count = len(white_high)
white_successes = white_high.sum()

for race in df_clean['race'].unique():
    if race != 'White':
        race_high = df_clean[df_clean['race'] == race]['high_income']
        race_count = len(race_high)
        race_successes = race_high.sum()
        
        if race_count > 0:
            # Two-proportion z-test
            count = [white_successes, race_successes]
            nobs = [white_count, race_count]
            z_stat, p_val = proportions_ztest(count, nobs)
            
            white_prop = white_successes / white_count
            race_prop = race_successes / race_count
            
            print(f"  {race} vs White: {race_prop:.1%} vs {white_prop:.1%}, "
                  f"z={z_stat:.3f}, p={p_val:.4f} "
                  f"{'*' if p_val < 0.05 else ''}")

# Test 5: Gender comparison
print(f"\nGender comparison:")
male_high = df_clean[df_clean['sex'] == 'Male']['high_income']
female_high = df_clean[df_clean['sex'] == 'Female']['high_income']

male_count = len(male_high)
female_count = len(female_high)
male_successes = male_high.sum()
female_successes = female_high.sum()

count = [male_successes, female_successes]
nobs = [male_count, female_count]
z_stat, p_val = proportions_ztest(count, nobs)

male_prop = male_successes / male_count
female_prop = female_successes / female_count

print(f"  Male vs Female: {male_prop:.1%} vs {female_prop:.1%}, "
      f"z={z_stat:.3f}, p={p_val:.2e} "
      f"{'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")

# 6. KEY FINDINGS SUMMARY
print(f"\n6. KEY FINDINGS SUMMARY:")
print("-" * 40)

race_disparity = race_high_income_summary.iloc[0]['High_Income_Rate'] - race_high_income_summary.iloc[-1]['High_Income_Rate']
sex_disparity = sex_high_income_summary.iloc[0]['High_Income_Rate'] - sex_high_income_summary.iloc[-1]['High_Income_Rate']
country_disparity = country_high_income_summary.iloc[0]['High_Income_Rate'] - country_high_income_summary.iloc[-1]['High_Income_Rate']

print(f"1. RACE: Largest disparity of {race_disparity:.1%} between {race_high_income_summary.index[0]} and {race_high_income_summary.index[-1]}")
print(f"2. SEX: Gender gap of {sex_disparity:.1%} between {sex_high_income_summary.index[0]} and {sex_high_income_summary.index[-1]}")
print(f"3. COUNTRY: Largest disparity of {country_disparity:.1%} between {country_high_income_summary.index[0]} and {country_high_income_summary.index[-1]}")

# Intersectional findings
race_sex_summary = df_clean.groupby(['race', 'sex'])['high_income'].agg(['count', 'mean']).round(4)
race_sex_summary.columns = ['Sample_Size', 'High_Income_Rate']
race_sex_summary = race_sex_summary.sort_values('High_Income_Rate', ascending=False)

highest_group = race_sex_summary.index[0]
lowest_group = race_sex_summary.index[-1]
highest_rate = race_sex_summary.iloc[0]['High_Income_Rate']
lowest_rate = race_sex_summary.iloc[-1]['High_Income_Rate']

print(f"4. INTERSECTIONAL: Highest earning group is {highest_group[0]} {highest_group[1]} ({highest_rate:.1%})")
print(f"   Lowest earning group is {lowest_group[0]} {lowest_group[1]} ({lowest_rate:.1%})")
print(f"   Intersectional disparity: {highest_rate - lowest_rate:.1%}")

# Create final summary table
print(f"\n7. SUMMARY TABLE: HIGH INCOME RATES BY DEMOGRAPHIC GROUP")
print("="*80)
summary_data = []

# Add race data
for race in df_clean['race'].unique():
    rate = df_clean[df_clean['race'] == race]['high_income'].mean()
    count = df_clean[df_clean['race'] == race].shape[0]
    summary_data.append(['Race', race, f"{rate:.1%}", f"{count:,}"])

# Add sex data
for sex in df_clean['sex'].unique():
    rate = df_clean[df_clean['sex'] == sex]['high_income'].mean()
    count = df_clean[df_clean['sex'] == sex].shape[0]
    summary_data.append(['Sex', sex, f"{rate:.1%}", f"{count:,}"])

# Add country data (top 5 by sample size)
for country in significant_countries[:5]:
    rate = df_countries[df_countries['native.country'] == country]['high_income'].mean()
    count = df_countries[df_countries['native.country'] == country].shape[0]
    summary_data.append(['Country', country, f"{rate:.1%}", f"{count:,}"])

summary_df = pd.DataFrame(summary_data, columns=['Category', 'Group', 'High_Income_Rate', 'Sample_Size'])
print(summary_df.to_string(index=False))

print(f"\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001")
print(f"\nAnalysis complete. Results saved to income_disparity_analysis.py")