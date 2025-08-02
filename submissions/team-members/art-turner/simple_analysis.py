#!/usr/bin/env python3
"""
Simple Income Disparity Analysis
Analysis of income disparities across race, sex, and native country in the Adult dataset
"""

import pandas as pd
import numpy as np

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

race_stats = df_clean.groupby('race')['high_income'].agg(['count', 'mean']).round(4)
race_stats.columns = ['Sample_Size', 'High_Income_Rate']
race_stats = race_stats.sort_values('High_Income_Rate', ascending=False)

for race, row in race_stats.iterrows():
    deviation = row['High_Income_Rate'] - overall_high_income_rate
    print(f"{race}: {row['High_Income_Rate']:.1%} ({deviation:+.1%} from average, n={row['Sample_Size']:,})")

# 2. SEX ANALYSIS
print(f"\n2. SEX DISPARITIES:")
print("-" * 40)

sex_stats = df_clean.groupby('sex')['high_income'].agg(['count', 'mean']).round(4)
sex_stats.columns = ['Sample_Size', 'High_Income_Rate']
sex_stats = sex_stats.sort_values('High_Income_Rate', ascending=False)

for sex, row in sex_stats.iterrows():
    deviation = row['High_Income_Rate'] - overall_high_income_rate
    print(f"{sex}: {row['High_Income_Rate']:.1%} ({deviation:+.1%} from average, n={row['Sample_Size']:,})")

# Calculate the gender pay gap
male_rate = sex_stats.loc['Male', 'High_Income_Rate']
female_rate = sex_stats.loc['Female', 'High_Income_Rate']
gender_gap = male_rate - female_rate

print(f"\nGender Income Gap Analysis:")
print(f"Male high income rate: {male_rate:.1%}")
print(f"Female high income rate: {female_rate:.1%}")
print(f"Gender gap: {gender_gap:.1f} percentage points")
print(f"Males are {male_rate/female_rate:.1f}x more likely to earn >$50K")

# 3. NATIVE COUNTRY ANALYSIS
print(f"\n3. NATIVE COUNTRY DISPARITIES:")
print("-" * 40)

# Focus on countries with meaningful sample sizes (>50 individuals)
country_counts = df_clean['native.country'].value_counts()
min_sample_size = 50
significant_countries = country_counts[country_counts >= min_sample_size].index.tolist()

print(f"Countries with >={min_sample_size} individuals: {len(significant_countries)}")
print("Top 10 countries by population:")
for i, (country, count) in enumerate(country_counts.head(10).items(), 1):
    print(f"{i:2d}. {country}: {count:,}")

# Filter dataset for significant countries
df_countries = df_clean[df_clean['native.country'].isin(significant_countries)].copy()

country_stats = df_countries.groupby('native.country')['high_income'].agg(['count', 'mean']).round(4)
country_stats.columns = ['Sample_Size', 'High_Income_Rate']
country_stats = country_stats.sort_values('High_Income_Rate', ascending=False)

print(f"\nHigh income rates by native country (countries with >={min_sample_size} individuals):")
for country, row in country_stats.iterrows():
    deviation = row['High_Income_Rate'] - overall_high_income_rate
    print(f"{country}: {row['High_Income_Rate']:.1%} ({deviation:+.1%} from average, n={row['Sample_Size']:,})")

# 4. INTERSECTIONAL ANALYSIS
print(f"\n4. INTERSECTIONAL ANALYSIS (Race and Sex):")
print("-" * 40)

race_sex_stats = df_clean.groupby(['race', 'sex'])['high_income'].agg(['count', 'mean']).round(4)
race_sex_stats.columns = ['Sample_Size', 'High_Income_Rate']
race_sex_stats = race_sex_stats.sort_values('High_Income_Rate', ascending=False)

print(f"High income rates by race and sex:")
for (race, sex), row in race_sex_stats.iterrows():
    print(f"{race} {sex}: {row['High_Income_Rate']:.1%} (n={row['Sample_Size']:,})")

# Calculate gender gaps within each race
print(f"\nGender gaps within each race:")
for race in df_clean['race'].unique():
    race_data = df_clean[df_clean['race'] == race]
    male_rate = race_data[race_data['sex'] == 'Male']['high_income'].mean()
    female_rate = race_data[race_data['sex'] == 'Female']['high_income'].mean()
    gap = male_rate - female_rate
    ratio = male_rate / female_rate if female_rate > 0 else float('inf')
    print(f"{race}: {gap:.1f} percentage points gap, {ratio:.1f}x ratio")

# 5. KEY FINDINGS SUMMARY
print(f"\n5. KEY FINDINGS SUMMARY:")
print("-" * 40)

race_disparity = race_stats.iloc[0]['High_Income_Rate'] - race_stats.iloc[-1]['High_Income_Rate']
sex_disparity = sex_stats.iloc[0]['High_Income_Rate'] - sex_stats.iloc[-1]['High_Income_Rate']
country_disparity = country_stats.iloc[0]['High_Income_Rate'] - country_stats.iloc[-1]['High_Income_Rate']

print(f"1. RACE: Largest disparity of {race_disparity:.1%} between {race_stats.index[0]} and {race_stats.index[-1]}")
print(f"2. SEX: Gender gap of {sex_disparity:.1%} between {sex_stats.index[0]} and {sex_stats.index[-1]}")
print(f"3. COUNTRY: Largest disparity of {country_disparity:.1%} between {country_stats.index[0]} and {country_stats.index[-1]}")

# Intersectional findings
highest_group = race_sex_stats.index[0]
lowest_group = race_sex_stats.index[-1]
highest_rate = race_sex_stats.iloc[0]['High_Income_Rate']
lowest_rate = race_sex_stats.iloc[-1]['High_Income_Rate']

print(f"4. INTERSECTIONAL: Highest earning group is {highest_group[0]} {highest_group[1]} ({highest_rate:.1%})")
print(f"   Lowest earning group is {lowest_group[0]} {lowest_group[1]} ({lowest_rate:.1%})")
print(f"   Intersectional disparity: {highest_rate - lowest_rate:.1%}")

# Create final summary table
print(f"\n6. SUMMARY TABLE: HIGH INCOME RATES BY DEMOGRAPHIC GROUP")
print("="*80)
print(f"{'Category':<10} {'Group':<20} {'High Income Rate':<15} {'Sample Size':<15}")
print("-" * 80)

# Add race data
for race in df_clean['race'].unique():
    rate = df_clean[df_clean['race'] == race]['high_income'].mean()
    count = df_clean[df_clean['race'] == race].shape[0]
    print(f"{'Race':<10} {race:<20} {rate:.1%:<15} {count:,}")

# Add sex data
for sex in df_clean['sex'].unique():
    rate = df_clean[df_clean['sex'] == sex]['high_income'].mean()
    count = df_clean[df_clean['sex'] == sex].shape[0]
    print(f"{'Sex':<10} {sex:<20} {rate:.1%:<15} {count:,}")

# Add country data (top 5 by sample size)
for country in significant_countries[:5]:
    rate = df_countries[df_countries['native.country'] == country]['high_income'].mean()
    count = df_countries[df_countries['native.country'] == country].shape[0]
    print(f"{'Country':<10} {country:<20} {rate:.1%:<15} {count:,}")

print(f"\nAnalysis complete.")