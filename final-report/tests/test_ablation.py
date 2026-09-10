import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_ablation.py"
SPEC = importlib.util.spec_from_file_location("run_ablation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mean_std_uses_sample_standard_deviation():
    result = MODULE.mean_std([1.0, 2.0, 3.0])
    assert result == {"mean": 2.0, "sd": 1.0}


def test_decision_requires_two_gains_and_guardrail():
    def row(margin, variation, active, agreement):
        return {key: {"mean": value} for key, value in {
            "top1_margin": margin, "spatial_variation": variation,
            "all_experts_active_rate": active, "appearance_agreement": agreement,
        }.items()}
    aggregate = {"mot": {"pretrained_init": row(0.10, 0.10, 0.30, 0.95),
                          "trained_10ep": row(0.13, 0.11, 0.45, 0.92)}}
    rule = {"criteria": {"top1_margin_relative_gain": 0.25, "spatial_variation_relative_gain": 0.25,
                           "all_experts_active_absolute_gain": 0.10}, "minimum_criteria_met": 2,
            "maximum_appearance_agreement_drop": 0.05}
    result = MODULE.evaluate_decision(aggregate, rule, "mot")
    assert result["criteria_met_count"] == 2
    assert result["supports_clearer_specialization"] is True


def test_guardrail_can_veto_routing_gains():
    def row(margin, variation, active, agreement):
        return {key: {"mean": value} for key, value in {
            "top1_margin": margin, "spatial_variation": variation,
            "all_experts_active_rate": active, "appearance_agreement": agreement,
        }.items()}
    aggregate = {"moa": {"pretrained_init": row(0.10, 0.10, 0.3, 0.98),
                          "trained_10ep": row(0.20, 0.20, 0.5, 0.80)}}
    rule = {"criteria": {"top1_margin_relative_gain": 0.25, "spatial_variation_relative_gain": 0.25,
                           "all_experts_active_absolute_gain": 0.10}, "minimum_criteria_met": 2,
            "maximum_appearance_agreement_drop": 0.05}
    assert MODULE.evaluate_decision(aggregate, rule, "moa")["supports_clearer_specialization"] is False


def test_zero_baseline_uses_positive_change_without_infinite_json_value():
    def row(margin, variation, active, agreement):
        return {key: {"mean": value} for key, value in {
            "top1_margin": margin, "spatial_variation": variation,
            "all_experts_active_rate": active, "appearance_agreement": agreement,
        }.items()}
    aggregate = {"mot": {"pretrained_init": row(0.2, 0.0, 0.0, 1.0),
                          "trained_10ep": row(0.2, 0.1, 0.2, 0.98)}}
    rule = {"criteria": {"top1_margin_relative_gain": 0.25, "spatial_variation_relative_gain": 0.25,
                           "all_experts_active_absolute_gain": 0.10}, "minimum_criteria_met": 2,
            "maximum_appearance_agreement_drop": 0.05}
    result = MODULE.evaluate_decision(aggregate, rule, "mot")
    assert result["gains"]["spatial_variation_relative_gain"] is None
    assert result["supports_clearer_specialization"] is True
