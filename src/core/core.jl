# Core: shared inference helpers, formula construction, result interface, panels.

include("inference.jl")
include("lp.jl")
include("formulas.jl")
include("estimate.jl")
include("data_structures.jl")
include("utils.jl")

export CausalEstimate, DiagnosticTest, WaldTest
export estimate, estimand, method_name, pvalues, tstats, rejects
export tidy, glance
export critical_value, two_sided_pvalue, wald_test, permutation_pvalue
export make_formula
export TreatmentPanel, validate_panel, preprocess_panel
