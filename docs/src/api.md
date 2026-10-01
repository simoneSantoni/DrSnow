# API Reference

```@meta
CurrentModule = DrSnow
```

One reference page per area; each documents every exported function and type of that area.

- [Core interface](reference/core.md)
- [Difference-in-Differences](reference/did.md)
- [Instrumental Variables](reference/iv.md)
- [Regression Discontinuity](reference/rd.md)
- [Synthetic Control](reference/synth.md)
- [Randomization Inference](reference/ri.md)
- [Sequential Inference](reference/sequential.md)
- [Adaptive Experiments](reference/adaptive.md)
- [Experimental Design and Power](reference/design.md)
- [Interference (SUTVA)](reference/sutva.md)
- [Causal Machine Learning](reference/ml.md)
- [Results and Plotting](reference/results.md)
- [Graphical Interface](reference/gui.md)

Accessors re-exported from StatsAPI (`coef`, `vcov`, `stderror`, `confint`, `coeftable`, `coefnames`, `nobs`, `dof_residual`, `pvalue`) work on every DrSnow result; see the [core interface](reference/core.md).

## Citation

To cite DrSnow in publications, use:

```bibtex
@software{drsnow,
  author = {Santoni, Simone},
  title  = {DrSnow: Design-Based Causal Inference for Natural Experiments in Julia},
  year   = {2026},
  url    = {https://github.com/simoneSantoni/DrSnow},
  note   = {Version 0.2}
}
```

Please also cite the papers behind the methods you use; each guide lists them.

## Index

```@index
Pages = ["reference/core.md", "reference/did.md", "reference/iv.md", "reference/rd.md", "reference/synth.md", "reference/ri.md", "reference/sequential.md", "reference/adaptive.md", "reference/design.md", "reference/sutva.md", "reference/ml.md", "reference/results.md", "reference/gui.md"]
```
