# Key Papers on SUTVA in Natural Experiments

This document provides specific paper recommendations organized by methodology and application area.

## 1. Foundational Papers

### SUTVA Framework

- **Rubin, D.B.** (1980). "Comment: Randomization Analysis of Experimental Data: The Fisher Randomization Test." *Journal of the American Statistical Association*
  - Original formulation of SUTVA in the Rubin Causal Model

- **Imbens, G.W. & Rubin, D.B.** (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*
  - Chapter 1: Comprehensive treatment of SUTVA and its implications

## 2. Network Interference (2023-2024)

### Approximate Neighborhood Interference

- **Leung, M.P.** (2024). "Causal Inference Under Approximate Neighborhood Interference" (arXiv:2212.xxxxx, updated Dec 2024)
  - **Key Contribution**: Relaxes no-interference to allow decaying effects with network distance
  - **Method**: IPW estimators remain consistent under ANI with specific network conditions
  - **Original**: *Econometrica* (2022)

- **Lu, X., et al.** (2024). "Adjusting auxiliary variables under approximate neighborhood interference" (arXiv:2024.xxxxx)
  - **Key Contribution**: Regression adjustment for network experiments
  - **Method**: Network-based covariate balancing for improved precision

### Partial Interference

- **Qu, Z., et al.** (2024). "Semiparametric Estimation of Treatment Effects in Observational Studies with Heterogeneous Partial Interference" (arXiv preprint, revised June 2024)
  - **Key Contribution**: Extends partial interference to observational studies
  - **Method**: Semiparametric approach allowing heterogeneous spillover patterns

- **Munro, E., et al.** (2023). "Causal Inference under Interference through Designed Markets"
  - **Key Contribution**: Addresses limitations of partial interference assumption
  - **Application**: Market design contexts

### Low-Order Network Interactions

- **Paper** (2023, *Journal of Causal Inference*): "Exploiting Neighborhood Interference with Low Order Interactions under Unit Randomized Design"
  - **Key Contribution**: Unbiased TTE estimation when effects constrained to low-order neighbor interactions
  - **Method**: Bernoulli randomized design with bounded-degree networks
  - **arXiv**: 2022, revised Feb 2024

## 3. Difference-in-Differences with Spillovers

### Spillover-Robust Methods

- **Working Paper** (2024, LMU Munich): "Spillover-Robust DiD Methods"
  - **Key Contribution**: Tests for local spillovers between clusters
  - **Method**: Flexible multi-dimensional distance metrics
  - **URL**: uni-muenchen.de

### AIPW for DiD with Interference

- **Yale Working Paper** (2024): "Doubly Robust DiD Estimators under Interference"
  - **Key Contribution**: Modified AIPW for DiD models with spillovers
  - **Method**: Doubly robust approach requiring either propensity or outcome model

### Place-Based Policy Decomposition

- **Burton, W.** (Recent): "DiD Decomposition for Place-Based Policies with Spillovers"
  - **Key Contribution**: Decomposes DiD into autarky, spillover-on-switchers, contamination effects
  - **Application**: Regional economic policies
  - **URL**: williamburton.eu

## 4. Experimental Design

### Two-Stage Randomization

- **arXiv papers** (2023-2024): Various papers on two-stage randomization for interference
  - **Design**: Cluster randomization → within-cluster individual randomization
  - **Estimands**: Separate direct and spillover effects

### Mixed Randomization

- **arXiv paper** (2024): "Mixed Randomization Designs for Interference"
  - **Design**: Combines Bernoulli and cluster-based randomization
  - **Goal**: Minimize bias in total treatment effect estimation

## 5. Spatial Methods

### Spatial Autoregressive Extensions

- **arXiv paper** (2024): "SAR Models for Synthetic Control with Spillovers"
  - **Key Contribution**: Extends SCM to account for spatial correlation
  - **Method**: Spatial autoregressive modeling

### Switchback Tests

- **Medium/Tech Blog** (2024): "Switchback Tests in Two-Sided Markets"
  - **Application**: Food delivery, ride-sharing platforms
  - **Design**: Time-unit or time-region randomization
  - **Goal**: Minimize crossover effects in dynamic systems

## 6. Natural Experiments - General

### Strengthening Causal Inference

- **NIH/PubMed** (2023): "Strengthening Causal Inference in Natural Experiment Studies"
  - **Recommendations**: Design features for robust natural experiments
  - **Elements**: Time-series data, multiple comparison groups, replication

### Learning Health Systems

- **NIH/PubMed** (2024): "Natural Experiments in Learning Health Systems"
  - **Application**: Real-world healthcare interventions
  - **Integration**: Electronic health records for causal inference

### Public Health Applications

- **Frontiers in Public Health** (2023): "Natural Experiments as Alternative to Clinical Trials"
  - **Context**: Community-based interventions
  - **Advantages**: Scalability, real-world validity

## 7. Computational Methods

### Algorithmic Solutions

- **arXiv** (2023): "Causal inference under interference: computational barriers and algorithmic solutions"
  - **Problem**: Unknown interference structures
  - **Solutions**: Computational approaches for partial interference

### Network-Based Deep Learning

- **ResearchGate/arXiv** (2024): "Graph Neural Networks for Causal Inference with Network Interference"
  - **Method**: GNN + instrumental variables
  - **Goal**: Mitigate hidden confounder bias in networks

## 8. Observational Studies

### Sensitivity Analysis

- **Various sources**: Papers on sensitivity analysis for SUTVA violations
  - **Approach**: Robustness checks under varying interference assumptions
  - **Output**: Bounds on treatment effects

### Bounding Methods

- **Manski's Work**: Referenced in Harvard lectures
  - **Approach**: Partial identification under social interactions
  - **Limitation**: Bounds can be wide/uninformative

## 9. Application-Specific

### Vaccine Studies

- **NIH/PubMed**: Multiple papers on spillover effects in vaccination programs
  - **Issue**: Herd immunity violates SUTVA
  - **Solution**: Two-stage designs, spillover modeling

### Education Interventions

- **Harvard/Kennedy School**: Papers on peer effects in education
  - **Issue**: Treated students influence untreated peers
  - **Solutions**: School-level clustering, network models

### Environmental Policy

- **Various**: Spatial spillovers in environmental interventions
  - **Issue**: Pollution crosses boundaries
  - **Solutions**: Spatial econometric methods

## 10. Software and Implementation

### R Packages

- **inferference**: Causal inference under interference
- **netdiffuseR**: Network diffusion and spillovers
- **CausalImpact**: Bayesian structural time-series (limited interference handling)

### Python Packages

- **DoWhy**: Causal inference library (network extensions)
- **NetworkX**: Network analysis (combined with causal inference)

## 11. Key Researchers to Follow

### Leading Scholars

- **Michael P. Leung** (UC Santa Cruz): Network interference, ANI
- **Zhaonan Qu**: Partial interference in observational studies
- **Various DiD researchers**: Recent extensions to spillover contexts

### Institutions

- **NBER**: Working papers on causal inference with interference
- **arXiv Economics/Statistics**: Pre-prints of cutting-edge methods
- **Harvard Kennedy School**: Applied natural experiments
- **Yale Economics**: Methodological developments

## 12. Access Recommendations

### Open Access Sources

1. **arXiv.org** - Latest pre-prints (economics, statistics sections)
2. **NBER Working Papers** - Some open access
3. **ResearchGate** - Researcher-uploaded papers
4. **SSRN** - Social science pre-prints

### Journal Focus

- *Econometrica* - Top-tier theoretical methods
- *Journal of Causal Inference* - Specialized methodological journal  
- *Quantitative Economics* - Applied econometrics
- *Epidemiology* - Public health applications
- *Political Analysis* - Social science applications

## 13. Suggested Reading Order

### For Beginners

1. Educational resources on SUTVA basics (Wikipedia, textbooks)
2. Leung (2024) - ANI framework overview
3. Simple two-stage randomization papers

### For Intermediate

1. Qu et al. (2024) - Partial interference
2. DiD spillover-robust methods
3. Network experiment design papers

### For Advanced

1. Endogenous interference models
2. Computational methods for unknown interference
3. GNN + IV approaches

---

## Notes on Citations

Many of the papers above are based on the web search results. For precise citations:

- Check arXiv for paper numbers and versions
- Verify journal publication status (some may still be working papers)
- Use Google Scholar for complete bibliographic information
- Access university repositories for working papers

**Last Updated**: Based on searches conducted December 2025
