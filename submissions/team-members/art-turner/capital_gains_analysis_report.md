# Capital Gains and Losses Analysis Report
## Impact on Income Classification in the Adult Dataset

### Executive Summary

This comprehensive analysis examines how capital gains and capital losses affect income classification (>$50K vs ≤$50K) in the Adult dataset. The findings reveal that **capital gains and losses are strong predictors of high income**, with significant statistical and practical implications.

### Key Findings

#### 1. Dataset Overview
- **Total sample size**: 32,561 individuals
- **High income rate**: 24.1% (>$50K)
- **Low income rate**: 75.9% (≤$50K)

#### 2. Capital Gains Distribution
- **Mean capital gains**: $1,077.65
- **Median capital gains**: $0.00
- **People with non-zero capital gains**: 2,712 (8.3%)
- **People with zero capital gains**: 29,849 (91.7%)

#### 3. Capital Losses Distribution
- **Mean capital losses**: $87.30
- **Median capital losses**: $0.00
- **People with non-zero capital losses**: 1,519 (4.7%)
- **People with zero capital losses**: 31,042 (95.3%)

#### 4. Income Group Differences

**Capital Gains:**
- High income group mean: $4,006.14
- Low income group mean: $148.75
- **Ratio (High/Low): 26.9x**

**Capital Losses:**
- High income group mean: $195.00
- Low income group mean: $53.14
- **Ratio (High/Low): 3.7x**

#### 5. Predictive Power Analysis

**Capital Gains:**
- P(High Income | Has Capital Gains) = **61.8%**
- P(High Income | No Capital Gains) = **20.7%**
- **Lift = 3.0x**

**Capital Losses:**
- P(High Income | Has Capital Losses) = **50.9%**
- P(High Income | No Capital Losses) = **22.8%**
- **Lift = 2.2x**

**Combined Analysis:**
- P(High Income | Has Either Gains or Losses) = **57.9%**
- P(High Income | Has Neither Gains nor Losses) = **19.0%**
- **Baseline probability**: 24.1%

#### 6. Statistical Significance

All relationships are highly statistically significant (p < 0.001):
- Capital Gains - Income correlation: **0.2233**
- Capital Losses - Income correlation: **0.1505**
- Has Capital Gains - Income correlation: **0.2662**
- Has Capital Losses - Income correlation: **0.1387**

#### 7. Effect Sizes

- Capital Gains Cohen's d: **0.5358 (medium effect)**
- Capital Losses Cohen's d: **0.3561 (small effect)**
- Capital Gains Cramer's V: **0.2660**
- Capital Losses Cramer's V: **0.1385**

### Non-Zero Capital Gains/Losses by Income Group

**Capital Gains:**
- Low income group: 4.2% have capital gains
- High income group: 21.4% have capital gains
- **Ratio: 5.1x more likely**

**Capital Losses:**
- Low income group: 3.0% have capital losses
- High income group: 9.9% have capital losses
- **Ratio: 3.3x more likely**

### Statistical Tests Results

1. **Chi-square Tests**: Both capital gains and losses show significant associations with income (p < 0.001)
2. **Mann-Whitney U Tests**: Significant differences in distributions between income groups
3. **Odds Ratios**: 
   - Capital gains: 6.23 (high income individuals are 6.23x more likely to have capital gains)
   - Capital losses: 3.51 (high income individuals are 3.51x more likely to have capital losses)

### Key Conclusions

1. **Capital gains are STRONG predictors of high income**
   - Having capital gains increases probability of high income by **37.8 percentage points**
   - 21.4% of high earners have capital gains vs 4.2% of low earners

2. **Capital losses are also significant predictors of high income**
   - Having capital losses increases probability of high income by **26.8 percentage points**
   - 9.9% of high earners have capital losses vs 3.0% of low earners

3. **Both features are statistically significant** with p-values < 0.001

4. **Effect sizes are meaningful**:
   - Capital gains show a medium effect size
   - Capital losses show a small effect size

5. **The majority of people have zero capital gains/losses**:
   - 91.7% have zero capital gains
   - 95.3% have zero capital losses

### Practical Implications

1. **Investment Activity Indicator**: Capital gains/losses are strong indicators of investment activity
2. **Income Association**: Investment activity is strongly associated with higher income
3. **Binary vs Continuous**: Having ANY capital gains/losses (even small amounts) is highly predictive
4. **Model Value**: These features should be valuable in income prediction models
5. **Feature Engineering**: The binary indicators (has/doesn't have) may be more useful than continuous amounts

### Recommendations

Based on this analysis, **capital gains and losses are STRONG predictors of income level** and should be included in any income prediction model. The binary indicators (presence/absence) are particularly powerful predictors.

**Specific recommendations:**
1. Include both capital gains and losses as features in income prediction models
2. Consider creating binary features (has_capital_gains, has_capital_losses) in addition to continuous amounts
3. Given the high percentage of zeros, consider specialized handling for zero-inflated distributions
4. The predictive power is substantial enough to warrant these features in any income classification task

### File Locations

- **Main Analysis Script**: `C:\Users\arttu\SDS-CP034-incomeinsight\submissions\team-members\art-turner\capital_gains_analysis.py`
- **Jupyter Notebook**: `C:\Users\arttu\SDS-CP034-incomeinsight\submissions\team-members\art-turner\adult_eda.ipynb`
- **Data File**: `C:\Users\arttu\SDS-CP034-incomeinsight\submissions\team-members\art-turner\data\adult.csv`
- **This Report**: `C:\Users\arttu\SDS-CP034-incomeinsight\submissions\team-members\art-turner\capital_gains_analysis_report.md`

---

*Analysis completed using Python with pandas, numpy, scipy, matplotlib, and seaborn libraries.*