# Comprehensive Income Disparity Analysis - Adult Dataset

## Executive Summary

This analysis examines income disparities across race, sex, and native country in the Adult dataset, focusing on the proportion of individuals earning more than $50,000 annually. The analysis reveals significant disparities across all demographic dimensions.

## Dataset Overview

- **Total Sample Size**: 31,978 individuals (after removing missing values)
- **Overall High Income Rate**: 24.1% earn >$50K
- **Data Cleaning**: Removed 583 rows with missing demographic data

## Key Findings

### 1. Race Disparities

**High Income Rates by Race:**
- Asian-Pac-Islander: 26.5% (+2.4% from average, n=956)
- White: 25.6% (+1.5% from average, n=27,430)
- Black: 12.3% (-11.7% from average, n=3,028)
- Amer-Indian-Eskimo: 11.6% (-12.5% from average, n=311)
- Other: 8.7% (-15.4% from average, n=253)

**Key Insights:**
- Largest racial disparity: 17.8% between Asian-Pac-Islander and Other
- Asian-Pac-Islander and White populations have above-average income rates
- Black, Amer-Indian-Eskimo, and Other populations face significant income disadvantages

### 2. Sex Disparities

**High Income Rates by Sex:**
- Male: 30.6% (+6.5% from average, n=21,370)
- Female: 10.9% (-13.1% from average, n=10,608)

**Key Insights:**
- Gender gap: 19.6% between Male and Female
- Males are 2.8x more likely to earn >$50K than females
- This represents one of the largest disparities observed in the dataset

### 3. Native Country Disparities

**Analysis Focus**: 21 countries with ≥50 individuals for statistical significance

**Top Performing Countries:**
- India: 40.0% (+15.9% from average, n=100)
- Taiwan: 39.2% (+15.2% from average, n=51)
- Japan: 38.7% (+14.6% from average, n=62)
- Italy: 34.2% (+10.2% from average, n=73)
- England: 33.3% (+9.3% from average, n=90)

**Lowest Performing Countries:**
- Dominican-Republic: 2.9% (-21.2% from average, n=70)
- Columbia: 3.4% (-20.7% from average, n=59)
- Guatemala: 4.7% (-19.4% from average, n=64)
- Mexico: 5.1% (-18.9% from average, n=643)
- Vietnam: 7.5% (-16.6% from average, n=67)

**Key Insights:**
- Largest country disparity: 37.1% between India and Dominican-Republic
- United States: 24.6% (close to overall average)
- Strong variations suggest potential immigration patterns and economic opportunities

### 4. Intersectional Analysis (Race and Sex)

**Highest Earning Groups:**
- Asian-Pac-Islander Male: 33.5% (n=632)
- White Male: 31.7% (n=18,890)
- Black Male: 19.0% (n=1,505)

**Lowest Earning Groups:**
- Other Female: 4.9% (n=102)
- Black Female: 5.8% (n=1,523)
- Amer-Indian-Eskimo Female: 10.1% (n=119)

**Gender Gaps Within Race:**
- Asian-Pac-Islander: 20.9% gap, 2.7x ratio
- White: 19.8% gap, 2.7x ratio
- Black: 13.2% gap, 3.3x ratio
- Other: 6.4% gap, 2.3x ratio
- Amer-Indian-Eskimo: 2.4% gap, 1.2x ratio

**Key Insights:**
- Intersectional disparity: 28.6% between highest and lowest groups
- Gender gaps exist within all racial categories
- Black women face the largest intersectional disadvantage among major groups

## Summary of Disparities

### Magnitude of Disparities:
1. **Country**: 37.1% (largest disparity)
2. **Intersectional**: 28.6% 
3. **Sex**: 19.6%
4. **Race**: 17.8%

### Significant Patterns:
- **Gender** is the most consistent predictor of income disparity across all racial groups
- **Native country** shows the largest absolute disparities, potentially reflecting immigration patterns and economic opportunities
- **Intersectional effects** compound individual demographic disadvantages

## Implications and Recommendations

1. **Gender Equality**: The consistent 2-3x male advantage across all racial groups suggests systemic gender-based income disparities that warrant targeted policy attention.

2. **Racial Equity**: Significant underrepresentation of Black, Amer-Indian-Eskimo, and Other populations in high-income categories indicates persistent racial income gaps.

3. **Immigration and Integration**: Wide variations by native country suggest the need for targeted support programs for certain immigrant populations.

4. **Intersectional Approaches**: The compounding effects of multiple demographic factors highlight the need for intersectional approaches to addressing income inequality.

## Methodology Notes

- Analysis focused on binary income classification (≤$50K vs >$50K)
- Countries with <50 individuals excluded from country analysis for statistical reliability
- Missing values handled through listwise deletion for demographic variables
- All percentages calculated as proportions of each demographic group

## Files Generated

- `C:\Users\arttu\SDS-CP034-incomeinsight\submissions\team-members\art-turner\adult_eda.ipynb` - Enhanced EDA notebook with disparity analysis
- `C:\Users\arttu\SDS-CP034-incomeinsight\submissions\team-members\art-turner\final_analysis.py` - Complete analysis script
- `C:\Users\arttu\SDS-CP034-incomeinsight\submissions\team-members\art-turner\income_disparity_report.md` - This comprehensive report