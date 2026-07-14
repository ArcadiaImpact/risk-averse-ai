# Prompt sets

Constitution-adjacent assets: prompt sets a character can be distilled or
evaluated against, kept beside the constitutions (mirroring aligne's
`character/` layout, where prompt sets sit next to the constitution JSONs). A
constitution's optional `default_prompts` field names one of these sets.

| file | aligne source | notes |
|------|---------------|-------|
| `risk_seeds.jsonl` | `src/aligne/character/prompts/risk_seeds.jsonl` (aligne `main`) | 56 general risk-tradeoff seed rollout prompts. The distill rollouts train on this set; the benchmark's gamble format is **held out** — never trained on. |

The canonical home is aligne; edit only by re-vendoring.
