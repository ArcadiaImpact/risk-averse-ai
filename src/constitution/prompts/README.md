# Prompt sets

Constitution-adjacent assets: prompt sets a character can be distilled or
evaluated against, kept beside the constitutions (mirroring aligne's
`character/` layout, where prompt sets sit next to the constitution JSONs). A
constitution's optional `default_prompts` field names one of these sets.

| file | aligne source | notes |
|------|---------------|-------|
| `risk_seeds.jsonl` | `src/aligne/character/prompts/risk_seeds.jsonl` (aligne `main`) | 56 general risk-tradeoff seed rollout prompts. The distill rollouts train on this set; the benchmark's gamble format is **held out** — never trained on. |
| `sft_prompts.jsonl` | — (extracted from the benchmark, see below) | 1,000 benchmark-format gamble-menu prompts (~half verbal-probability), the `prompt_text` column of the low-stakes CoT training CSV. Used by the **matched-prompts** distill arm to hold the prompt distribution fixed at the SFT training set while swapping only the supervision signal. |

`risk_seeds.jsonl`'s canonical home is aligne; edit only by re-vendoring.

## `sft_prompts.jsonl` provenance

Extracted by `experiments/constitution-distill/scripts/extract_sft_prompts.py`
(idempotent; run order preserved) from the `prompt_text` column of
`src/third_party/riskaverseAIs/sft-training/data/CoT-training/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv`
— the benchmark's designated *training* split. **Only the prompts are lifted**:
no `chosen_full`/`rejected_full` demonstrations and no answer-key columns are
read, so this arm trains on the benchmark's training-split *prompts* but never
its responses, and never touches the val/test/deployment splits. This is the
deliberate third held-out category (see `CLAUDE.md`).
