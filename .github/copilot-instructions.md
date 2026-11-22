# Project Context

This repository is a completed computer vision research project titled
"Flood Segmentation: Graph Cut vs U-Net vs SegFormer."

The project compares three binary flood segmentation methods:
1. Classical: Graph Cut
2. From scratch: U-Net
3. Pretrained modern: SegFormer-B0

The project uses FloodNet plus Flood Area Segmentation and Roadway Flooding,
with additional local Philippine flood photos for testing.

## Documentation / Research Rules

- Treat the repository as the source of truth for implementation details and results.
- Never invent, estimate, or assume experimental results.
- Before making factual claims about results, inspect the relevant files in
  `results/`, experiment scripts, saved CSVs, and generated figures.
- Prefer actual CSV values and generated outputs over README summaries when both exist.
- If a result cannot be verified from the repository, explicitly say that it
  could not be verified.
- Keep report writing academic, clear, concise, and factual.
- Do not exaggerate model performance or claim statistical significance unless
  it was actually tested.
- Distinguish clearly between measured results, observations, and interpretation.
- Preserve the project's terminology and methodology.

## Results

Important result files include:
- `results/main_results.csv`
- `results/per_image_test.csv`
- `results/ablation_data_size.csv`
- `results/efficiency.csv`
- `results/domain_shift.csv`
- `results/failure_per_class.csv`
- `results/training_runs.csv`

Relevant generated figures include:
- `results/*.png`
- `curves_*.png`
- `classical_tuning_*.png`
- `sanity_classical*.png`

When asked for results, inspect the relevant CSV/data first and use the
actual values.

## Visualization

When creating graphs for the report:
- Use Python and matplotlib unless the repository already uses another plotting library.
- Make publication/report-ready figures.
- Include clear titles, axis labels, units, legends, and readable fonts.
- Use consistent model names: Graph Cut, U-Net, SegFormer-B0.
- Do not distort axes or omit relevant data to make a model look better.
- Save figures under an appropriate `results/` subdirectory.
- Prefer simple bar charts for model comparisons and line charts for training
  curves or data-size experiments.
- Never fabricate missing values.

## Code Safety

The core project implementation is already complete.
Do not unnecessarily rewrite or refactor working model code.
For documentation and analysis tasks, prefer read-only inspection and
separate analysis/visualization scripts.

## Reproducibility

When possible, identify:
- the source file containing the result,
- the experiment that generated it,
- the relevant metric,
- and the exact data used.

If a requested analysis requires rerunning an experiment, say so before doing
so and use the existing scripts rather than creating a parallel implementation.