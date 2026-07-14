# No-think-tag low-stakes copies

These CSVs are copies of the current March 22 low-stakes training sets with the Qwen-specific `<think>` and `</think>` tags removed from `chosen_full` and `rejected_full`.

Why these exist:
- The reasoning text and final answer JSON are preserved.
- Only the outer `<think>` wrappers were removed.
- These copies are used for Llama and Gemma runs, which do not use the Qwen-style think tags.

Files:
- `2026_03_22_low_stakes_training_set_1000_situations_CoTs_no_think_tags.csv` from `2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv`
- `2026_04_13_tie_training_modified_CoTs_no_think_tags.csv` from `2026_04_13_tie_training_modified_CoTs.csv`
