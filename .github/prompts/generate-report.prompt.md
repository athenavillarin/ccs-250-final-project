# Generate Final Report

Generate the complete final report for this computer vision project.

## Required structure

1. Introduction
2. Methods
3. Experiments
4. Results
5. Analysis and Discussion
6. Conclusion
7. References
8. Contribution Statement

## Instructions

Before writing, inspect the repository and identify:
- project README
- source code for all three methods
- dataset and split definitions
- experiment scripts
- `results/*.csv`
- existing result figures
- training histories
- relevant documentation

The report must describe the project as it actually exists in this repository.

### Introduction
Explain:
- flood segmentation problem
- motivation
- project objective
- three approaches being compared
- research question

### Methods
Document:
- datasets and their sizes
- preprocessing
- train/validation/test splits
- binary flood-mask definition
- Graph Cut implementation
- U-Net architecture and training
- SegFormer-B0 and fine-tuning
- hardware/software where supported by the repository

### Experiments
Explain every experiment actually performed:
- main model comparison
- data-size ablation
- efficiency comparison
- domain-shift experiment
- failure analysis

For each experiment, state its purpose, setup, and metrics.

### Results
Use the ACTUAL values from the repository.

Before writing numerical results:
- inspect the relevant CSV files directly
- do not rely only on README summaries
- do not invent missing values
- do not round values unnecessarily
- preserve appropriate units

Include useful tables and identify existing figures that should appear in the report.

If an important visualization does not exist:
- create a report-ready Python/matplotlib figure
- save it under `results/figures/`
- create or update a small reproducible script for it

Recommended figures include:
- main model performance comparison
- runtime / parameter / training-time comparison
- training curves
- data-size ablation
- domain-shift comparison
- failure analysis

### Analysis and Discussion
Do not merely repeat the numbers.

Explain:
- which method performed best and according to which metric
- trade-offs between quality and efficiency
- differences between classical, from-scratch, and pretrained approaches
- effects of training-data size
- effects of domain shift
- important failure patterns

Only make explanations supported by the implementation, results, or established
properties of the methods. Clearly distinguish measured findings from interpretation.

### Conclusion
Directly answer the research question using the experimental evidence.

### Writing requirements

- Formal undergraduate research-paper style
- Clear and concise
- No marketing language
- No unsupported claims
- No fabricated results
- No "the results prove" unless actually justified
- Use consistent names: Graph Cut, U-Net, SegFormer-B0
- Use the project's actual terminology
- Do not rewrite or modify working model code
- Do not omit experiments simply because their results are inconvenient
- Flag any information that cannot be verified from the repository

## Important

The repository is the source of truth for this report.

For every major numerical claim, know which repository file supports it.

At the end, provide a short "Report Evidence Map" listing:
- section
- important claim/result
- source file