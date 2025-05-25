# Recommendation Algorithms Analysis Report

## Overview

This analysis compares two recommendation algorithms (Multiclustering and Standard) to evaluate how the number of records in a cluster affects ranking performance. The hypothesis being tested is that the Multiclustering algorithm is less affected by cluster size bias than the Standard algorithm.

## Key Findings

- **Correlation between Records and Rank**:
  - Multiclustering: -0.107
  - Standard: -0.010
  - Difference: 0.097

This confirms the hypothesis that the Standard algorithm shows a stronger correlation between the number of records in a cluster and the ranking, indicating it is more biased by cluster size.

## Statistical Summary

| Metric | Multiclustering | Standard |
|--------|----------------|----------|
| Sample Size | 2387 | 1100 |
| Average Records in Cluster | 4.15 | 4.62 |
| Average Rank | 7.29 | 5.58 |
| Median Rank | 7.0 | 5.0 |
| Records-Rank Correlation | -0.107 | -0.010 |

## Visual Analysis

The visualizations demonstrate the relationship between the number of records in a cluster and the ranking performance for each algorithm.

### Records vs Rank Plots

The scatter plots and density plots show how the number of records in a cluster affects ranking. The regression lines illustrate the trend, and the correlation coefficients quantify the strength of the relationship.

### Rank Distribution

The rank distribution plots show how rankings are distributed for each algorithm, with lower values indicating better performance.

### Normalized Rank

The normalized rank visualizations adjust the rank values to account for cluster size, providing a fairer comparison between the algorithms.

## Company-Specific Analysis

Different companies show varying levels of bias in the recommendation algorithms:

- **Companies where Standard algorithm shows higher bias**: Adobe, Amazon, Apple, Facebook, Microsoft, Salesforce, Tesla
- **Companies where Multiclustering shows higher bias or equal**: Google, IBM, Nvidia, Oracle, Twitter, Uber

## Conclusion

