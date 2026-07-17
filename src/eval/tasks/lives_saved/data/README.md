# lives_saved transfer benchmark

`2026_04_11_lives_saved_transfer_benchmark_interleaved_1000_situations.csv` is the
April 11, 2026 lives-saved transfer set: the gamble situations recast so the
stake is lives saved rather than dollars, interleaved across all four stakes
levels (250 low / 250 medium / 250 high / 250 astronomical, in that repeating
order) for 1000 situations total. Each row carries the transfer provenance
columns `source_stakes`, `source_condition`, `source_csv_name`,
`source_situation_id`. Bound to the `lives_saved_transfer_benchmark` dataset
alias and scored like the other gamble tasks (see the shared recipe note in
`../../gpu_hours/data/README.md`).
