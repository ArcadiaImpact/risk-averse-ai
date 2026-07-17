# money_for_user transfer benchmark

`2026_04_11_money_for_user_transfer_benchmark_interleaved_1000_situations.csv` is
the April 11, 2026 money-for-user transfer set: the gamble situations recast so
the stake is money accruing to the user rather than the assistant, interleaved
across all four stakes levels (250 low / 250 medium / 250 high / 250
astronomical, in that repeating order) for 1000 situations total. Each row
carries the transfer provenance columns `source_stakes`, `source_condition`,
`source_csv_name`, `source_situation_id`. Bound to the
`money_for_user_transfer_benchmark` dataset alias and scored like the other
gamble tasks (see the shared recipe note in `../../gpu_hours/data/README.md`).
