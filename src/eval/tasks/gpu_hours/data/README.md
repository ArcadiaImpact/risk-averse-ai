# Transfer-To-Other-Quantities Benchmarks

These CSVs interleave the April 11, 2026 transfer-to-other-quantities source sets across all four stakes levels:

- `low_stakes_training`
- `medium_stakes_validation`
- `high_stakes_test`
- `astronomical_stakes_deployment`

For each treatment condition, the combined benchmark contains `1000` situations total:

- `250` low-stakes situations
- `250` medium-stakes situations
- `250` high-stakes situations
- `250` astronomical-stakes situations

Files:

- `2026_04_11_gpu_hours_transfer_benchmark_interleaved_1000_situations.csv`
- `2026_04_11_lives_saved_transfer_benchmark_interleaved_1000_situations.csv`
- `2026_04_11_money_for_user_transfer_benchmark_interleaved_1000_situations.csv`

The rows are interleaved by stakes in the order:

1. low
2. medium
3. high
4. astronomical

and then repeated.

Extra provenance columns added to each row:

- `source_stakes`
- `source_condition`
- `source_csv_name`
- `source_situation_id`

The canonical evaluator (`evaluate.py`) has dataset aliases for these three combined CSVs and now logs expected-value transfer metrics per situation and in the top-level summary JSON, including:

- whether the chosen option is an EV-maximizer
- whether the chosen option is an EV-minimizer
- chosen EV divided by the best EV in the situation
- chosen EV normalized to the available EV range
- EV regret (`best EV - chosen EV`)
