import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from risk_averse_prompts import (
    CLI_SYSTEM_PROMPT_SOURCE,
    DATASET_DEFAULT_SYSTEM_PROMPT_SOURCE,
    MODEL_DEFAULT_NO_SYSTEM_PROMPT_SOURCE,
    TRANSFER_QUANTITY_SYSTEM_PROMPT,
    default_system_prompt_for_dataset,
    model_uses_no_system_prompt,
    resolve_system_prompt,
)


class RiskAversePromptResolutionTests(unittest.TestCase):
    def test_transfer_dataset_uses_transfer_prompt_for_non_gemma_models(self):
        prompt, source = resolve_system_prompt(
            dataset_base_alias="gpu_hours_transfer_benchmark",
            base_model="Qwen/Qwen3-8B",
            explicit_system_prompt=None,
        )
        self.assertEqual(prompt, TRANSFER_QUANTITY_SYSTEM_PROMPT)
        self.assertEqual(source, DATASET_DEFAULT_SYSTEM_PROMPT_SOURCE)

    def test_gemma_3_12b_defaults_to_no_system_prompt(self):
        prompt, source = resolve_system_prompt(
            dataset_base_alias="medium_stakes_validation",
            base_model="google/gemma-3-12b-it",
            explicit_system_prompt=None,
        )
        self.assertEqual(prompt, "")
        self.assertEqual(source, MODEL_DEFAULT_NO_SYSTEM_PROMPT_SOURCE)

    def test_explicit_system_prompt_overrides_gemma_default(self):
        prompt, source = resolve_system_prompt(
            dataset_base_alias="medium_stakes_validation",
            base_model="google/gemma-3-12b-it",
            explicit_system_prompt="Use this custom prompt",
        )
        self.assertEqual(prompt, "Use this custom prompt")
        self.assertEqual(source, CLI_SYSTEM_PROMPT_SOURCE)

    def test_model_name_detection_matches_gemma_adapters_too(self):
        self.assertTrue(model_uses_no_system_prompt("/tmp/gemma-3-12b-it_seed1_adapter"))
        self.assertTrue(model_uses_no_system_prompt("meta-llama/Llama-3.1-8B-Instruct"))
        self.assertFalse(model_uses_no_system_prompt("Qwen/Qwen3-8B"))

    def test_non_transfer_dataset_keeps_normal_default_for_non_gemma_models(self):
        prompt, source = resolve_system_prompt(
            dataset_base_alias="medium_stakes_validation",
            base_model="Qwen/Qwen3-8B",
            explicit_system_prompt=None,
        )
        self.assertEqual(prompt, default_system_prompt_for_dataset("medium_stakes_validation"))
        self.assertEqual(source, DATASET_DEFAULT_SYSTEM_PROMPT_SOURCE)


if __name__ == "__main__":
    unittest.main()
