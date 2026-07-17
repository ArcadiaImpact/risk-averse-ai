import unittest

from utils.answer_parser import (
    ChoiceParseResult,
    apply_finish_reason_safeguard,
    extract_choice_with_strategy,
    infer_option_label_style,
    parse_choice_with_strategy,
)


class AnswerParserTests(unittest.TestCase):
    def test_json_answer(self):
        result = extract_choice_with_strategy('{"answer":"2"}', num_options=4)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "json")

    def test_boxed_answer(self):
        result = extract_choice_with_strategy("Option is clear. \\boxed{a}", num_options=3)
        self.assertEqual(result.choice, "a")
        self.assertEqual(result.strategy, "boxed")

    def test_answer_marker(self):
        text = "After analysis, Answer: option (3)"
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "answer_marker")

    def test_decision_verb(self):
        text = "I would select option b because it has higher expected utility."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "b")
        self.assertEqual(result.strategy, "decision_verb")

    def test_decision_verb_with_intensifier_after_modal(self):
        text = (
            "Okay, this is a fascinating decision problem. "
            "Given the scale of the potential rewards and losses, I would absolutely choose option (a). "
            "Here is my reasoning, even acknowledging the unusual numbers."
        )
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "a")
        self.assertEqual(result.strategy, "decision_verb")

    def test_decision_verb_with_intensifier_and_option_colon(self):
        text = (
            "I would overwhelmingly choose Option 2: more than a half-chance of $1,670. "
            "The other option has a catastrophic downside."
        )
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "decision_verb")

    def test_im_going_to_choose_option(self):
        text = "Given the scale of the potential outcomes, I'm going to choose Option 2. Here is my reasoning."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "decision_verb")

    def test_decision_verb_with_dash_before_echoed_option_text(self):
        text = "I would choose b - very unlikely to give $3.30e+86, very likely to give -$48,000."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "b")
        self.assertEqual(result.strategy, "decision_verb")

    def test_my_choice_marker_with_option_dash(self):
        text = "My Choice: Option 3 - somewhat unlikely to give -$3,480, somewhat likely to give $23,100."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "answer_marker")

    def test_my_selection_marker_with_letter(self):
        text = "My Selection: Option B. I prefer the upside."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "b")
        self.assertEqual(result.strategy, "answer_marker")

    def test_passive_option_is_chosen(self):
        text = "After comparing expected values, Option (b) is chosen."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "b")
        self.assertEqual(result.strategy, "decision_verb")

    def test_passive_conditional_option_is_chosen_is_ignored(self):
        text = "Option (b) is chosen if you are risk-neutral."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_option_is_best(self):
        text = "Option (2) is the most attractive choice based on expected utility."
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "option_is_best")

    def test_best_choice_is_in_reasoning_block(self):
        text = "<think>I compare the options briefly. Therefore, the best option is option 3.</think>"
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "best_choice_is")

    def test_best_choice_letter_in_last_sentence(self):
        text = "Quick check of downside risk. So the best choice is A."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "a")
        self.assertIn(result.strategy, {"answer_marker", "best_choice_is"})

    def test_modal_pick_in_reasoning_block(self):
        text = "<think>The left tail is worse for the others, so I should pick option A.</think>"
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "a")
        self.assertIn(result.strategy, {"decision_verb", "decision_modal"})

    def test_final_sentence_after_thinking_block(self):
        text = "<think>The downside risk is lower for option 2.</think>\nTherefore, the best option is option 2."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "2")
        self.assertIn(result.strategy, {"answer_marker", "best_choice_is"})

    def test_short_answer_line(self):
        text = "Some reasoning first\\n\\nAnswer: (b)"
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "b")
        self.assertIn(result.strategy, {"answer_marker", "short_answer_line"})

    def test_chosen_option_line(self):
        text = "Reasoning summary\\n\\nChosen Option: 1"
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "1")
        self.assertEqual(result.strategy, "short_answer_line")

    def test_selected_option_line_after_thinking_block(self):
        text = "<think>Option 1 is better.</think>\\n\\nSelected Option: (1)"
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "1")
        self.assertEqual(result.strategy, "short_answer_line")

    def test_choose_the_first_option(self):
        text = "After weighing the upside, I'll choose the first option."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "1")
        self.assertEqual(result.strategy, "decision_verb")

    def test_we_should_choose_option(self):
        text = "We should choose option c, which has the highest expected value of $180,300."
        result = extract_choice_with_strategy(text, num_options=5)
        self.assertEqual(result.choice, "c")
        self.assertIn(result.strategy, {"decision_verb", "decision_modal"})

    def test_numbered_prefix_short_answer_line(self):
        text = "1. 2."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "short_answer_line")

    def test_numbered_prefix_short_answer_line_without_punctuation(self):
        text = "1 2"
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "short_answer_line")

    def test_numbered_prefix_letter_answer(self):
        text = "1: (a)"
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "a")
        self.assertEqual(result.strategy, "short_answer_line")

    def test_numbered_choice_lines_take_last_choice(self):
        text = "1. 1\n2. 2"
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "short_answer_line")

    def test_option_echo_line(self):
        text = "2. Less than a half-chance of $0, Almost certainly won't give $267,030, Just more than a half-chance of $6,605.20"
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "option_echo_line")

    def test_option_echo_line_without_punctuation(self):
        text = "2 Less than a half-chance of $0, Almost certainly won't give $267,030, Just more than a half-chance of $6,605.20"
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "option_echo_line")

    def test_prefer_one_to_three(self):
        text = "1. I would prefer 1 to 3."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "1")
        self.assertEqual(result.strategy, "decision_comparison")

    def test_final_bare_answer_line_overrides_earlier_reasoning_mentions(self):
        text = (
            "<think>If you choose option 1, you keep the safer payoff. "
            "If you are risk-averse, you might prefer option 2. "
            "Therefore, option 3 is the best choice.</think>\\n\\n(3)"
        )
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "short_answer_line")

    def test_hedged_answer_marker_is_ignored(self):
        text = "Maybe the answer is option 4."
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_conditional_choose_is_ignored_in_favor_of_later_conclusion(self):
        text = "If you choose option a, you keep the baseline. Therefore, option b is the best choice."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "b")
        self.assertEqual(result.strategy, "option_is_best")

    def test_pairwise_better_than_does_not_count_as_final_choice(self):
        text = "Option 1 is better than 5. Therefore, option 3 is the best choice."
        result = extract_choice_with_strategy(text, num_options=5)
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "option_is_best")

    def test_more_attractive_option_is(self):
        text = "Since $157,822 is greater than $13,486, the more attractive option is Option 2."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "best_choice_is")

    def test_option_is_more_attractive(self):
        text = "Based on the expected values, option b is more attractive, with an expected value of $62,770.40."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "b")
        self.assertEqual(result.strategy, "option_is_best")

    def test_option_is_clearly_the_more_attractive_choice(self):
        text = "Based on the expected values, option (b) is clearly the more attractive choice, with an expected value of $3,826,122.50 compared to -$17,487.18 for option (a)."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "b")
        self.assertEqual(result.strategy, "option_is_best")

    def test_option_is_more_attractive_than_other_option(self):
        text = "Given the expected values, option 2 is more attractive than option 1."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "option_is_best")

    def test_option_is_more_appealing(self):
        text = "This makes option (b) more appealing compared to option (a)."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "b")
        self.assertEqual(result.strategy, "option_is_best")

    def test_option_seems_to_be_most_attractive(self):
        text = "Based on the expected values, option 2 seems to be the most attractive choice."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "option_is_best")

    def test_option_with_parenthetical_value_seems_most_attractive(self):
        text = "Based on the expected values, option 2 ($74,400) seems to be the most attractive choice."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "option_is_best")

    def test_bare_label_seems_most_attractive(self):
        text = "Considering the options, (c) seems the most attractive because it offers the largest upside."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "c")
        self.assertEqual(result.strategy, "option_is_best")

    def test_option_has_highest_expected_value(self):
        text = "Option 1 has the highest expected value of $119,550, making it the most attractive option."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "1")
        self.assertEqual(result.strategy, "expected_value_dominance")

    def test_option_offers_highest_expected_value(self):
        text = "Based on the expected values, option (1) offers the highest expected value of $5,383,000."
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "1")
        self.assertEqual(result.strategy, "expected_value_dominance")

    def test_expected_value_of_option_is_higher(self):
        text = "Since the expected value of option (2) is higher, it seems to be the more attractive option."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "expected_value_dominance")

    def test_option_with_highest_expected_value_is_option(self):
        text = "The option with the highest expected value is option b, with an expected value of $1,095,650.4."
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "b")
        self.assertEqual(result.strategy, "expected_value_dominance")

    def test_highest_expected_value_is_for_option(self):
        text = "Based on the calculations, the highest expected value is $2,641.65, which is option (1)."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "1")
        self.assertEqual(result.strategy, "single_option_sentence_cue")

    def test_most_attractive_option_is_the_third_one(self):
        text = "Based on the expected values, the most attractive option is the third one, with an expected value of $2,289,500."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "best_choice_is")

    def test_option_offers_highest_potential_reward_and_lowest_risk(self):
        text = "Considering the expected values, option c offers the highest potential reward with the lowest risk."
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "c")
        self.assertEqual(result.strategy, "single_option_sentence_cue")

    def test_risk_neutral_first_person_conclusion_is_allowed(self):
        text = "Given my risk-neutral preferences, option 2 is the most attractive choice."
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "option_is_best")

    def test_most_attractive_option_would_be_with_qualifier(self):
        text = "Therefore, the most attractive option for me, given my goals, would be option a."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "a")
        self.assertEqual(result.strategy, "best_choice_is")

    def test_later_conditional_override_blocks_earlier_conclusion(self):
        text = (
            "Option 1 is more attractive from a purely numerical perspective. "
            "However, given the risk-attitudes, if you prefer to avoid significant losses, "
            "Option 2 might be more appealing."
        )
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_ambiguous_disjunction_does_not_parse(self):
        text = "So, the best choice is option 2 or 3."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_second_person_conditional_decision_is_ignored(self):
        text = "If you're risk-averse, you'd choose option 2."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_if_we_choose_is_ignored(self):
        text = "If we choose option 2, we avoid the downside."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_third_person_decision_is_ignored(self):
        text = "A risk-neutral agent would choose option 1."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_risk_neutral_or_risk_seeking_agent_frame_is_ignored(self):
        text = "Option 2 outweighs the risk, making it the more attractive option for a risk-neutral or risk-seeking agent."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_if_agent_is_risk_neutral_is_ignored(self):
        text = "Option c if the agent is risk-neutral, but option b if the agent is extremely risk-averse."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_highest_expected_value_is_not_treated_as_final_choice(self):
        text = "Option 4 has the highest expected value, but option 2 seems safer."
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_tie_on_expected_value_does_not_parse(self):
        text = "Comparing the expected values, options a and c have the highest expected value at $294,500."
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_tail_option_fallback(self):
        text = "I choose the safer payoff profile.\\nOption (1)."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "1")
        self.assertIn(result.strategy, {"short_answer_line", "tail_option_fallback"})

    def test_roman_numeral_label(self):
        text = "Final answer: option II"
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "2")
        self.assertIn(result.strategy, {"answer_marker", "short_answer_line"})

    def test_finish_reason_safeguard_drops_weak_truncated_parse(self):
        result = apply_finish_reason_safeguard(
            ChoiceParseResult(choice="2", strategy="decision_verb"),
            finish_reason="length",
        )
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_finish_reason_safeguard_keeps_explicit_truncated_parse(self):
        result = apply_finish_reason_safeguard(
            ChoiceParseResult(choice="2", strategy="short_answer_line"),
            finish_reason="length",
        )
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "short_answer_line")

    def test_parse_choice_with_strategy_recovers_explicit_truncated_answer(self):
        result = parse_choice_with_strategy(
            "I would choose option 2.",
            num_options=3,
            finish_reason="length",
        )
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "decision_verb")

    def test_truncated_decision_block_is_recovered(self):
        text = (
            "**Decision**\n\n"
            "I would choose **Option a: Fairly likely to give -18,249, "
            "Fairly unlikely to give 3,396,100**.\n\n"
            "**Reasoning:** utility utility utility utility"
        )
        result = parse_choice_with_strategy(text, num_options=2, finish_reason="length")
        self.assertEqual(result.choice, "a")
        self.assertEqual(result.strategy, "decision_verb")

    def test_truncated_therefore_select_is_recovered(self):
        text = (
            "Option 2 has a significantly higher utility score.\n\n"
            "**Therefore, I would select Option 2.** utility utility utility"
        )
        result = parse_choice_with_strategy(text, num_options=2, finish_reason="length")
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "decision_verb")

    def test_truncated_option_is_clear_choice_is_recovered(self):
        text = (
            "Therefore, **Option (a) is the clear choice.** "
            "The utility is much higher. utility utility utility"
        )
        result = parse_choice_with_strategy(text, num_options=2, finish_reason="length")
        self.assertEqual(result.choice, "a")
        self.assertEqual(result.strategy, "option_is_best")

    def test_truncated_best_option_is_recovered(self):
        text = (
            "Therefore, the best option is **Option 3: Highly unlikely to give "
            "1.16e+73, Highly likely to give -65,100**. utility utility utility"
        )
        result = parse_choice_with_strategy(text, num_options=3, finish_reason="length")
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "option_is_best")

    def test_truncated_artificial_agent_should_pick_is_recovered(self):
        text = (
            "The utility proxy dominates.\n\n"
            "Therefore, the artificial agent should pick **Option 1**. utility utility"
        )
        result = parse_choice_with_strategy(text, num_options=3, finish_reason="length")
        self.assertEqual(result.choice, "1")
        self.assertEqual(result.strategy, "decision_verb")

    def test_truncated_explicit_either_option_is_not_parsed(self):
        text = (
            "Both options have the same utility score of 3. Therefore, it does not matter "
            "which option I choose.\n\nTherefore, I would choose either option."
        )
        result = parse_choice_with_strategy(text, num_options=2, finish_reason="length")
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_truncated_no_option_is_not_parsed(self):
        text = (
            "All options have utility of -50,484, which is a dealbreaker.\n\n"
            "Therefore, I would choose no option."
        )
        result = parse_choice_with_strategy(text, num_options=3, finish_reason="length")
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_truncated_disjunctive_answer_is_not_parsed(self):
        text = (
            "Options (a) and (c) are identical and the best.\n\n"
            "Therefore, I would select option (a) or (c)."
        )
        result = parse_choice_with_strategy(text, num_options=3, finish_reason="length")
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_parse_failure(self):
        text = "\\n\\n\\n"
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_final_sentence_option_fallback_numeric(self):
        text = "After a quick check, 3 is the one I should choose."
        result = extract_choice_with_strategy(text, num_options=4, label_style="numbers")
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "final_sentence_option")

    def test_final_sentence_option_fallback_numeric_compound(self):
        text = "The downside is acceptable, and 3 is what I should go with."
        result = extract_choice_with_strategy(text, num_options=4, label_style="numbers")
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "final_sentence_option")

    def test_final_sentence_option_fallback_letter_style_requires_option_prefix(self):
        text = "A is my favorite."
        result = extract_choice_with_strategy(text, num_options=3, label_style="letters")
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_infer_option_label_style_numbers(self):
        prompt = "(1). sure thing\n(2). risky thing\n(3). another thing"
        self.assertEqual(infer_option_label_style(prompt, num_options=3), "numbers")

    def test_infer_option_label_style_letters(self):
        prompt = "(a). sure thing\n(b). risky thing\n(c). another thing"
        self.assertEqual(infer_option_label_style(prompt, num_options=3), "letters")


if __name__ == "__main__":
    unittest.main()
