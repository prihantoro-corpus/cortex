"""
Quick diagnostic script to check why all adjectives show 0 effect size.
Tests the pseudo-sample approach with example data.
"""
import numpy as np
from scipy import stats

# Example: word "cantik" appears 45 times in group f, 8 times in group m
freq1 = 45
freq2 = 8

# Current approach: pseudo-samples
sample1 = np.ones(int(freq1))
sample2 = np.ones(int(freq2)) * 2

print("=== Current Pseudo-Sample Approach ===")
print(f"Sample 1 (f): {sample1[:5]}... (mean={sample1.mean()}, std={sample1.std()})")
print(f"Sample 2 (m): {sample2[:5]}... (mean={sample2.mean()}, std={sample2.std()})")
print()

# Run t-test
t_stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=False)
print(f"T-test result: t={t_stat:.3f}, p={p_val:.6f}")

# Run Mann-Whitney
u_stat, p_val_mw = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
print(f"Mann-Whitney result: U={u_stat:.3f}, p={p_val_mw:.6f}")

# Cohen's h
total = freq1 + freq2
p1 = freq1 / total
p2 = freq2 / total
h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
print(f"Cohen's h: {h:.3f}")
print()

# The issue: samples have NO variance within groups!
print("PROBLEM: Pseudo-samples have zero variance within each group")
print("This doesn't properly model count data differences")
print()

# Better approach for count data: Chi-square or proportion test
print("=== Better Approach: Chi-Square Test ===")
# 2x2 contingency table: [freq in group, total_group_size - freq]
# But we don't know total group sizes...

# For MVP: Use simple proportion comparison
print(f"Proportion in f: {p1:.3f}, Proportion in m: {p2:.3f}")
print(f"Difference: {abs(p1-p2):.3f}")
print()

# Effect size interpretation
if abs(h) < 0.2:
    print("Effect size: SMALL (< 0.2)")
elif abs(h) < 0.5:
    print("Effect size: MEDIUM (0.2-0.5)")
else:
    print("Effect size: LARGE (> 0.5)")
