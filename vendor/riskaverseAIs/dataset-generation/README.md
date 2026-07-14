# Dataset Generation

This directory contains the scripts used to generate the paper's benchmark CSVs.

The paper's reported results use the pre-generated CSVs shipped in
`../evaluation/data/`. Re-running the generators is optional and mainly useful
for auditing the data-generation procedure described in the appendix.

The only dependencies are `pandas` and `numpy` (no GPU or model access needed).

Main scripts:

- `generate_main_policy_datasets.py`: medium-stakes validation, high-stakes test, astronomical-stakes deployment, and steals-test style datasets.
- `generate_low_stakes_training_dataset.py`: low-stakes training-set generator.
- `generate_gpu_hours_transfer_dataset.py`: transfer-domain datasets in the GPU-hours framing.
- `generate_lives_saved_transfer_dataset.py`: transfer-domain datasets in the lives-saved framing.
- `generate_money_for_user_transfer_dataset.py`: transfer-domain datasets in the money-for-user framing.

Each script is a direct, cleaned export of the corresponding generation
notebook. Adjust the configuration block at the top of the script before
running it.
