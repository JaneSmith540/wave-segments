import numpy as np
import pandas as pd

from wave_segments.class_robustness import perturb_features_from_train_scale, probability_total_variation


def test_feature_perturbation_is_seeded_train_scaled_and_holds_structural_fields_fixed():
    train = pd.DataFrame({"x": np.arange(101, dtype=float), "duration_bars": np.arange(101) + 1,
                          "boundary_uncertainty": np.linspace(0, 1, 101)})
    scored = train.iloc[:5].copy()
    first = perturb_features_from_train_scale(scored, train, list(train.columns), noise_fraction=.2, random_state=9)
    second = perturb_features_from_train_scale(scored, train, list(train.columns), noise_fraction=.2, random_state=9)
    assert first.equals(second)
    assert first.duration_bars.equals(scored.duration_bars)
    assert first.boundary_uncertainty.equals(scored.boundary_uncertainty)
    assert not first.x.equals(scored.x)
    assert scored.equals(train.iloc[:5])  # no in-place mutation


def test_probability_total_variation_is_zero_for_equal_rows_and_bounded():
    left = pd.DataFrame({"prob_A": [.8, .5], "prob_B": [.2, .5]})
    right = pd.DataFrame({"prob_A": [.8, .1], "prob_B": [.2, .9]})
    values = probability_total_variation(left, right, ["prob_A", "prob_B"])
    assert values[0] == 0
    assert np.isclose(values[1], .4)
    assert ((values >= 0) & (values <= 1)).all()
