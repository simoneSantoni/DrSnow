using Documenter
using DrSnow

DocMeta.setdocmeta!(DrSnow, :DocTestSetup, :(using DrSnow); recursive=true)

makedocs(;
    modules = [DrSnow],
    authors = "Simone Santoni and contributors",
    repo = Remotes.GitHub("simoneSantoni", "DrSnow"),
    sitename = "DrSnow.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://simonesantoni.github.io/DrSnow/",
        assets = ["assets/custom.css"],
        sidebar_sitename = false,
        size_threshold_warn = 384 * 1024,
        size_threshold = 800 * 1024,
        search_size_threshold_warn = 3 * 1024 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Methods" => [
            "Difference-in-Differences" => "did.md",
            "Instrumental Variables" => ["iv.md", "iv_designs.md", "iv_ml.md"],
            "Regression Discontinuity" => "rd.md",
            "Synthetic Control" => "synth.md",
            "Randomization Inference" => "ri.md",
            "Sequential Inference" => "sequential.md",
            "Adaptive Experiments" => "adaptive.md",
            "Experimental Design and Power" => "design.md",
            "Interference (SUTVA)" => "sutva.md",
            "Causal Machine Learning" => ["ml.md", "ml_hte.md", "ml_measurement.md"],
        ],
        "Results and Plotting" => "results.md",
        "Graphical Interface" => "gui.md",
        "Validation" => "validation.md",
        "API Reference" => [
            "Overview" => "api.md",
            "Core interface" => "reference/core.md",
            "Difference-in-Differences" => "reference/did.md",
            "Instrumental Variables" => "reference/iv.md",
            "Regression Discontinuity" => "reference/rd.md",
            "Synthetic Control" => "reference/synth.md",
            "Randomization Inference" => "reference/ri.md",
            "Sequential Inference" => "reference/sequential.md",
            "Adaptive Experiments" => "reference/adaptive.md",
            "Experimental Design and Power" => "reference/design.md",
            "Interference (SUTVA)" => "reference/sutva.md",
            "Causal Machine Learning" => [
                "Learners and DML" => "reference/ml.md",
                "Heterogeneous effects and policy" => "reference/ml_hte.md",
                "ML-measured variables" => "reference/ml_measurement.md",
            ],
            "Results and Plotting" => "reference/results.md",
            "Graphical Interface" => "reference/gui.md",
        ],
    ],
    checkdocs = :exports,
)

# Pages publishes a single manual at the site root, without Documenter's deploydocs.
# Supply the metadata normally written by deploydocs to disable version switching.
write(joinpath(@__DIR__, "build", "siteinfo.js"),
      "var DOCUMENTER_VERSION_SELECTOR_DISABLED = true;\n")
