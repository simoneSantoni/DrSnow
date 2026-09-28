# Cutting-Edge Research on LATE Issues in Natural Experiments

> [!NOTE]
> This document provides a comprehensive overview of recent advances (2023-2024) in Local Average Treatment Effect (LATE) methodology and its application to natural experiments.

## Executive Summary

The Local Average Treatment Effect (LATE) framework, developed by Imbens and Angrist (1994), has become the cornerstone of causal inference in natural experiments. Recent research (2023-2024) addresses critical issues including:

- **Relaxing monotonicity assumptions** to allow for "defiers"
- **Heterogeneous treatment effects** and external validity concerns
- **Inference methods** robust to weak instruments and many covariates
- **Alternative identification strategies** that don't require traditional IV assumptions
- **Integration with machine learning** for discovering treatment effect heterogeneity

---

## 1. Key LATE Concepts and Foundations

### What is LATE?

**Local Average Treatment Effect (LATE)** estimates the average causal effect of a treatment for a specific subgroup called "compliers" — individuals whose treatment status is influenced by an instrumental variable (IV).

**Key distinction from ATE:**

- **ATE (Average Treatment Effect):** Effect for entire population
- **LATE:** Effect only for compliers (marginal individuals influenced by the instrument)

### Core Identifying Assumptions

1. **Relevance:** The instrument Z affects treatment D
2. **Exclusion restriction:** Z affects outcome Y only through D
3. **Independence:** Z is independent of potential outcomes and treatment
4. **Monotonicity:** No "defiers" (individuals who do the opposite of assignment)

### The Compliers Framework

Individuals are classified into four types:

- **Compliers:** Take treatment if assigned, don't if not assigned (LATE applies to this group)
- **Always-takers:** Always take treatment regardless of assignment
- **Never-takers:** Never take treatment regardless of assignment
- **Defiers:** Do the opposite of assignment (assumed not to exist under monotonicity)

---

## 2. Recent Methodological Advances (2023-2024)

### 2.1 Weakening the Monotonicity Assumption

> [!IMPORTANT]
> **"It is never too LATE: a new look at local average treatment effects with or without defiers"** (July 2023)

**Key contributions:**

- Explores LATE identification under **weaker monotonicity assumptions**
- Allows for the presence of "defiers" in certain contexts
- Proposes **new estimators** potentially more efficient than traditional 2SLS
- Includes empirical application to returns to education

**Implications:** Researchers can now apply LATE in settings where strict monotonicity is questionable.

**Citation:** Published in *Econometrica*, July 2023

---

### 2.2 Inference with Covariates and Treatment Effect Heterogeneity

> [!IMPORTANT]
> **"Inference on LATEs with covariates"** (February 2024)

**Key contributions:**

- Provides **asymptotically valid tests and confidence intervals** for weighted average LATEs
- Particularly robust in settings with:
  - Treatment effect heterogeneity
  - Many covariates
  - Weak identification strength

**Implications:** Addresses a critical gap in inference methods when treatment effects vary across observed characteristics.

**Citation:** Published February 2024, Singapore Management University

---

### 2.3 Two-Stage Inference for Randomized Experiments

> [!IMPORTANT]
> **Two-stage inference procedure for sample LATE** (September 2024)

**Key contributions:**

- Introduces inference procedure for randomized experiments with **non-compliance**
- Accounts for whether IV is **strong or weak**
- Provides valid inference in finite samples

**Citation:** Available on arXiv, September 2024

---

### 2.4 Marginal Treatment Effects Without IV Assumptions

> [!WARNING]
> This represents a **paradigm shift** in how we think about treatment effect heterogeneity

**"MTE without IV Assumptions"** (January & August 2024)

**Key contributions:**

- Proposes method for defining, identifying, and estimating MTEs **without traditional IV assumptions**
- Does not require:
  - Independence assumption
  - Exclusion restriction
  - Separability
- Defines MTE based on reduced-form treatment error statistically independent of covariates
- Provides means to **test validity of candidate IVs**

**Implications:** Particularly valuable when valid instruments are hard to find in natural experiments.

**Citations:** arXiv, January 2024 and August 2024

---

### 2.5 MTE Identification Using Subjective Expectations

**Key contributions:**

- Identifies individual latent propensity to select into treatment using **survey data on subjective expectations**
- Allows assessment of treatment effect heterogeneity
- Enables inference in **counterfactual policy environments**

**Citation:** NBER Working Paper, 2024

---

## 3. Critical Issues in Natural Experiments

### 3.1 Conceptual Design Issues

> [!CAUTION]
> **Critical examination of natural experiment designs** (2021, widely cited in 2023-2024)

**Key concerns:**

- Many natural experiments use IVs with only **indirect effects** on treatment
- This can lead to **biased or uninterpretable LATE estimates**
- Researchers must carefully define the treatment of interest

**Implication:** Not all "natural" variation provides valid instruments for LATE estimation.

**Citation:** RePEc discussion paper, 2021

---

### 3.2 External Validity of LATE Estimates

> [!WARNING]
> LATE estimates may not generalize beyond the complier subpopulation

**Key insights:**

- **Angrist and Fernandez-Val (2010)** introduced covariate-based approach to assess external validity
- Method uses **reweighting technique** to construct estimates for new subpopulations
- Uses overidentification test to define populations where IV estimates are externally valid

**Recent developments (2023-2024):**

- The "Credibility Revolution" emphasized internal validity but may have neglected external validity
- Growing recognition that **construct and external validity** are essential for generalized causal claims

**Practical implications:**

- LATE is specific to compliers whose treatment is affected by the instrument
- If treatment effects are heterogeneous, LATE ≠ ATE
- Researchers must be cautious when generalizing LATE findings

---

### 3.3 Heterogeneous Treatment Effects and LATE

**Key relationships:**

- When treatment effects vary across individuals, IV analysis measures effects for **marginal patients** (compliers)
- Standard IV estimators (2SLS) identify a **weighted average of LATEs**
- Weights depend on:
  - First-stage coefficients
  - Variability of covariates and instruments
  - First-stage effect heterogeneity

**Machine learning integration:**

- Recent work leverages ML techniques to **discover subgroups** with varying causal effects
- Helps estimate heterogeneous treatment effects using IVs
- Addresses unmeasured confounding in observational studies

---

## 4. Notable Applications and Examples

### 4.1 Oregon Health Insurance Experiment

Used extensively to illustrate:

- LATE vs. MTE comparisons
- Treatment effect heterogeneity across compliance types
- Causal mediation in natural experiments (forthcoming 2025)

### 4.2 Education and Returns to Schooling

- Classic application of LATE framework
- Recent papers examine robustness to monotonicity violations
- Tests new estimators against traditional 2SLS

### 4.3 Health Interventions

- Natural experiments in hepatology research (2023 review)
- Herpes zoster vaccination and dementia (NIH R01 grants, 2023-2024)
- Nature-based interventions for health outcomes

---

## 5. Practical Recommendations for Researchers

### When Designing Natural Experiments

1. **Carefully define the treatment of interest**
   - Ensure IV has direct effect on treatment, not just indirect associations

2. **Assess monotonicity plausibility**
   - Consider whether defiers could exist in your context
   - Use new methods (2023) if monotonicity is questionable

3. **Plan for heterogeneity**
   - Consider covariates that might moderate treatment effects
   - Use methods robust to treatment effect heterogeneity (2024 inference methods)

4. **Address external validity**
   - Characterize the complier population
   - Use reweighting techniques to assess generalizability
   - Be transparent about limitations

### When Analyzing Data

5. **Choose appropriate estimators**
   - Consider alternatives to 2SLS if more efficient options exist (2023 advances)
   - Use two-stage inference if dealing with potential weak instruments (2024)

6. **Test IV validity**
   - Use new MTE methods to test whether candidate IVs satisfy assumptions (2024)
   - Conduct sensitivity analyses for assumption violations

7. **Report comprehensively**
   - Characterize compliers using observed characteristics
   - Discuss external validity explicitly
   - Report both LATE and, if possible, heterogeneity across subgroups

---

## 6. Emerging Trends and Future Directions

### Integration with Machine Learning

- Discovering treatment effect heterogeneity
- Automated IV selection and validation
- Causal forest methods extended to IV settings

### Causal Mediation Analysis

- MTE-based approaches to understand mechanisms (forthcoming 2025)
- Decomposing total effects into direct and indirect paths

### Relaxing Assumptions

- Moving beyond strict monotonicity
- Alternative identification without traditional IV assumptions
- Leveraging subjective expectations data

### Application Domains

- Population health interventions
- Policy evaluation with natural variation
- Transdisciplinary natural experiments

---

## 7. Key Papers to Read

### Foundational Papers

1. **Imbens & Angrist (1994)** - "Identification and Estimation of Local Average Treatment Effects" *Econometrica*
2. **Angrist, Imbens & Rubin (1996)** - "Identification of Causal Effects Using Instrumental Variables" *JASA*

### Recent Methodological Advances (2023-2024)

3. **"It is never too LATE: a new look at local average treatment effects with or without defiers"** (July 2023) - *Econometrica*
4. **"Inference on LATEs with covariates"** (February 2024) - SMU
5. **"MTE without IV Assumptions"** (January & August 2024) - arXiv
6. **"Two-stage inference procedure for sample LATE"** (September 2024) - arXiv

### External Validity

7. **Angrist & Fernandez-Val (2010)** - Extrapolating IV estimates
8. **Recent critiques of natural experiment designs** (2021, RePEc)

### Applications

9. **Oregon Health Insurance Experiment** - Amy Finkelstein et al.
10. **Returns to education** - Various IV applications

---

## 8. Glossary

- **ATE:** Average Treatment Effect
- **LATE:** Local Average Treatment Effect
- **MTE:** Marginal Treatment Effect
- **IV:** Instrumental Variable
- **2SLS:** Two-Stage Least Squares
- **ITT:** Intention-to-Treat Effect
- **CACE:** Complier Average Causal Effect (synonym for LATE)
- **Compliers:** Individuals whose treatment status is affected by the instrument
- **Defiers:** Individuals who do the opposite of their assignment
- **Monotonicity:** Assumption that there are no defiers

---

## Conclusion

The LATE framework continues to evolve with cutting-edge research addressing its core assumptions and extending its applicability. Recent advances (2023-2024) have:

- **Relaxed strict assumptions** (monotonicity, IV requirements)
- **Improved inference methods** for complex settings
- **Enhanced external validity assessment**
- **Integrated modern computational methods**

For researchers using natural experiments, these advances provide more robust and flexible tools for causal inference, while also highlighting the importance of careful design and transparent reporting of limitations.

> [!TIP]
> Stay updated on this rapidly evolving literature by following:
>
> - NBER Working Paper series on econometrics
> - arXiv econometrics section (econ.EM)
> - Leading journals: *Econometrica*, *Journal of Econometrics*, *Review of Economics and Statistics*
