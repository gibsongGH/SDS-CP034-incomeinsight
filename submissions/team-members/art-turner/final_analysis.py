#!/usr/bin/env python3
"""
Final Income Disparity Analysis
Analysis of income disparities across race, sex, and native country in the Adult dataset
"""

import pandas as pd
import numpy as np

# Load the dataset
print("Loading Adult dataset...")
df = pd.read_csv('data/adult.csv')

# Data cleaning
df_clean = df.copy()
df_clean = df_clean.replace('?', np.nan)
key_vars = ['race', 'sex', 'native.country', 'income']
df_clean = df_clean.dropna(subset=key_vars)
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
print("-" * 50)

race_stats = df_clean.groupby('race')['high_income'].agg(['count', 'mean']).round(4)
race_stats.columns = ['Sample_Size', 'High_Income_Rate']
race_stats = race_stats.sort_values('High_Income_Rate', ascending=False)

for race, row in race_stats.iterrows():
    deviation = row['High_Income_Rate'] - overall_high_income_rate
    print(f"{race:18s}: {row['High_Income_Rate']:.1%} ({deviation:+.1%} from avg, n={row['Sample_Size']:,.0f})")

# 2. SEX ANALYSIS
print(f"\n2. SEX DISPARITIES:")
print("-" * 50)

sex_stats = df_clean.groupby('sex')['high_income'].agg(['count', 'mean']).round(4)
sex_stats.columns = ['Sample_Size', 'High_Income_Rate']
sex_stats = sex_stats.sort_values('High_Income_Rate', ascending=False)

for sex, row in sex_stats.iterrows():
    deviation = row['High_Income_Rate'] - overall_high_income_rate
    print(f"{sex:18s}: {row['High_Income_Rate']:.1%} ({deviation:+.1%} from avg, n={row['Sample_Size']:,.0f})")

# Gender gap analysis
male_rate = sex_stats.loc['Male', 'High_Income_Rate']
female_rate = sex_stats.loc['Female', 'High_Income_Rate']
gender_gap = male_rate - female_rate

print(f"\nGender Income Gap Analysis:")
print(f"  Male high income rate: {male_rate:.1%}")
print(f"  Female high income rate: {female_rate:.1%}")
print(f"  Gender gap: {gender_gap:.1%}")
print(f"  Males are {male_rate/female_rate:.1f}x more likely to earn >$50K")

# 3. NATIVE COUNTRY ANALYSIS
print(f"\n3. NATIVE COUNTRY DISPARITIES:")
print("-" * 50)

# Focus on countries with meaningful sample sizes
country_counts = df_clean['native.country'].value_counts()
min_sample_size = 50
significant_countries = country_counts[country_counts >= min_sample_size].index.tolist()

print(f"Analyzing {len(significant_countries)} countries with >={min_sample_size} individuals")
print("Top 10 countries by population:")
for i, (country, count) in enumerate(country_counts.head(10).items(), 1):
    print(f"  {i:2d}. {country}: {count:,}")

# Filter dataset for significant countries
df_countries = df_clean[df_clean['native.country'].isin(significant_countries)].copy()

country_stats = df_countries.groupby('native.country')['high_income'].agg(['count', 'mean']).round(4)
country_stats.columns = ['Sample_Size', 'High_Income_Rate']
country_stats = country_stats.sort_values('High_Income_Rate', ascending=False)

print(f"\nHigh income rates by native country:")
for country, row in country_stats.iterrows():
    deviation = row['High_Income_Rate'] - overall_high_income_rate
    print(f"{country:18s}: {row['High_Income_Rate']:.1%} ({deviation:+.1%} from avg, n={row['Sample_Size']:,.0f})")

# 4. INTERSECTIONAL ANALYSIS
print(f"\n4. INTERSECTIONAL ANALYSIS (Race and Sex):")
print("-" * 50)

race_sex_stats = df_clean.groupby(['race', 'sex'])['high_income'].agg(['count', 'mean']).round(4)
race_sex_stats.columns = ['Sample_Size', 'High_Income_Rate']
race_sex_stats = race_sex_stats.sort_values('High_Income_Rate', ascending=False)

print(f"High income rates by race and sex:")
for (race, sex), row in race_sex_stats.iterrows():
    print(f"{race} {sex:6s}: {row['High_Income_Rate']:.1%} (n={row['Sample_Size']:,.0f})")

# Gender gaps within each race
print(f"\nGender gaps within each race:")
for race in df_clean['race'].unique():
    race_data = df_clean[df_clean['race'] == race]
    male_rate = race_data[race_data['sex'] == 'Male']['high_income'].mean()
    female_rate = race_data[race_data['sex'] == 'Female']['high_income'].mean()
    gap = male_rate - female_rate
    ratio = male_rate / female_rate if female_rate > 0 else float('inf')
    print(f"{race:18s}: {gap:.1%} gap, {ratio:.1f}x ratio")

# 5. KEY FINDINGS SUMMARY
print(f"\n5. KEY FINDINGS SUMMARY:")
print("="*50)

race_disparity = race_stats.iloc[0]['High_Income_Rate'] - race_stats.iloc[-1]['High_Income_Rate']
sex_disparity = sex_stats.iloc[0]['High_Income_Rate'] - sex_stats.iloc[-1]['High_Income_Rate']
country_disparity = country_stats.iloc[0]['High_Income_Rate'] - country_stats.iloc[-1]['High_Income_Rate']

print(f"RACE DISPARITY: {race_disparity:.1%} between {race_stats.index[0]} and {race_stats.index[-1]}")
print(f"SEX DISPARITY: {sex_disparity:.1%} between {sex_stats.index[0]} and {sex_stats.index[-1]}")
print(f"COUNTRY DISPARITY: {country_disparity:.1%} between {country_stats.index[0]} and {country_stats.index[-1]}")

# Intersectional findings
highest_group = race_sex_stats.index[0]
lowest_group = race_sex_stats.index[-1]
highest_rate = race_sex_stats.iloc[0]['High_Income_Rate']
lowest_rate = race_sex_stats.iloc[-1]['High_Income_Rate']

print(f"INTERSECTIONAL DISPARITY: {highest_rate - lowest_rate:.1%}")
print(f"  Highest: {highest_group[0]} {highest_group[1]} ({highest_rate:.1%})")
print(f"  Lowest: {lowest_group[0]} {lowest_group[1]} ({lowest_rate:.1%})")

# 6. SUMMARY TABLE
print(f"\n6. SUMMARY TABLE:")
print("="*50)
print(f"{'Category':<10} {'Group':<18} {'Rate':<8} {'Sample':<8}")
print("-" * 50)

# Race data
race_ordered = race_stats.sort_values('High_Income_Rate', ascending=False)
for race, row in race_ordered.iterrows():
    print(f"{'Race':<10} {race:<18} {row['High_Income_Rate']:.1%:<8} {row['Sample_Size']:,.0f}")

# Sex data  
sex_ordered = sex_stats.sort_values('High_Income_Rate', ascending=False)
for sex, row in sex_ordered.iterrows():
    print(f"{'Sex':<10} {sex:<18} {row['High_Income_Rate']:.1%:<8} {row['Sample_Size']:,.0f}")

# Top 5 countries
print(f"{'Country':<10} {'(Top 5 by pop)':<18} {'Rate':<8} {'Sample':<8}")
for country in significant_countries[:5]:
    rate = df_countries[df_countries['native.country'] == country]['high_income'].mean()
    count = df_countries[df_countries['native.country'] == country].shape[0]
    print(f"{'Country':<10} {country:<18} {rate:.1%:<8} {count:,}")

print(f"\nAnalysis complete.")
print("="*80)