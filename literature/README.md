# DrSnow Literature Review Index

This directory contains comprehensive literature reviews organized by econometric methodology and topic area. Each subdirectory focuses on a specific area of causal inference research.

## Directory Structure

### 📊 Difference-in-Differences (DiD)

**Location:** [`did/`](did/)

Literature on DiD methods for causal inference with panel data and repeated cross-sections.

- [**Comprehensive Guide**](did/comprehensive_guide.md) - In-depth review of DiD methodology, covering traditional approaches, recent innovations, and practical implementation guidance
- [**Quick Reference**](did/quick_reference.md) - Concise reference guide for DiD methods, key papers, and common diagnostics
- [**Recent Literature (2024-2025)**](did/recent_2024_2025.md) - Cutting-edge papers and methodological advances from 2024-2025

---

### 🔄 Stable Unit Treatment Value Assumption (SUTVA)

**Location:** [`sutva/`](sutva/)

Research on SUTVA violations, spillover effects, and network interference in causal inference.

- [**Comprehensive Survey**](sutva/comprehensive_survey.md) - Detailed survey of SUTVA violation literature, detection methods, and remedies
- [**Key Papers**](sutva/key_papers.md) - Essential papers on SUTVA, interference effects, and spatial spillovers

---

### 🎯 Local Average Treatment Effect (LATE)

**Location:** [`late/`](late/)

Literature on instrumental variables, compliance, and LATE in natural experiments.

- [**Natural Experiments Summary**](late/natural_experiments_summary.md) - Key papers on LATE estimation in natural experiments, monotonicity assumptions, and external validity

---

### 📈 Heterogeneous Treatment Effects (HTE)

**Location:** [`hte/`](hte/)

Research on estimating and understanding treatment effect heterogeneity.

- [**Review**](hte/review.md) - Comprehensive review of HTE estimation methods, from traditional subgroup analysis to modern machine learning approaches

---

### 🤖 Causal Machine Learning

**Location:** [`causal_ml/`](causal_ml/)

Literature on combining machine learning with causal inference for observational data.

- [**Observational Data**](causal_ml/observational_data.md) - Recent advances in causal ML with observational data, including double/debiased ML, causal forests, and deep learning approaches

---

## Topic Cross-References

These methodologies often intersect and complement each other:

- **DiD + HTE**: Modern DiD methods for heterogeneous treatment effects (see [`did/comprehensive_guide.md`](did/comprehensive_guide.md))
- **SUTVA + Networks**: Network interference and spillovers (see [`sutva/comprehensive_survey.md`](sutva/comprehensive_survey.md))
- **LATE + HTE**: Heterogeneous treatment effects in IV/LATE frameworks (see [`late/natural_experiments_summary.md`](late/natural_experiments_summary.md))
- **Causal ML + All Topics**: ML methods apply across DiD, HTE, and other causal frameworks (see [`causal_ml/observational_data.md`](causal_ml/observational_data.md))

## Related Resources

- **DrSnow Package**: Implementation of these methods in Julia (see [`../src/`](../src/)
  and the methods guides in [`../docs/src/`](../docs/src/))
- **Examples**: Practical applications and tutorials (see [`../examples/`](../examples/))

---

*Last updated: 2025-12-13*
