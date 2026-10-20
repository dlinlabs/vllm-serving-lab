import copy

import numpy as np
import pytest

import diagnose_flow_rollout as diagnostic
from test_transition_model import synthetic_runs
from transition_baselines import BASE
from transition_features import FLOWS, STATES


class ConstantHead:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


class InputHead:
    def __init__(self, key):
        self.index = BASE.index(key)

    def predict(self, x):
        return x[:, self.index]


class Probe:
    engine = False

    def __init__(self, key='received_5s'):
        truth = synthetic_runs()['fixed-low']['rows'][0]['next_flows']
        self.heads = {k: ConstantHead(truth[k]) for k in FLOWS}
        self.reward = InputHead(key)


def fixture_rows():
    rows = copy.deepcopy(synthetic_runs()['fixed-low']['rows'])
    for i, row in enumerate(rows):
        row['features']['received_5s'] = i+1.
    return rows


def test_free_prediction_cannot_read_future_observations_or_flows():
    a = fixture_rows()
    b = copy.deepcopy(a)
    for row in b[1:5]:
        row['features'].update(received_5s=1e8, reserved_cost=1e8, engine_waiting=1e8)
        row['next_flows']['first'] = 1e7
        row['planned_caps'] = [1e8]*5
    # Labels remain unchanged: scoring truth is evidence, never predictor input.
    p, q = diagnostic.rollout(Probe(), a, 0, 'free'), diagnostic.rollout(Probe(), b, 0, 'free')
    assert [x['prediction'] for x in p] == [x['prediction'] for x in q]


def test_arrival_refresh_is_explicit_oracle_and_other_history_groups_do_not_refresh_it():
    rows = fixture_rows()
    for mode in ('free','state_reset','engine_refresh','activity_history_refresh','outcome_history_refresh'):
        assert diagnostic.rollout(Probe(),rows,0,mode)[-1]['prediction']['cohort_ttft_met']==5.
    for mode in ('arrival_history_refresh','history_refresh','all_observations_reset'):
        assert diagnostic.rollout(Probe(),rows,0,mode)[-1]['prediction']['cohort_ttft_met']==15.


def test_outcome_refresh_does_not_silently_use_arrival_or_activity_fields():
    rows = fixture_rows()
    for i,row in enumerate(rows):
        row['features']['resolved_ttft_met_5s'] = i+2.
    model = Probe('resolved_ttft_met_5s')
    assert diagnostic.rollout(model,rows,0,'arrival_history_refresh')[-1]['prediction']['cohort_ttft_met']==10.
    assert diagnostic.rollout(model,rows,0,'activity_history_refresh')[-1]['prediction']['cohort_ttft_met']==10.
    assert diagnostic.rollout(model,rows,0,'outcome_history_refresh')[-1]['prediction']['cohort_ttft_met']==20.


def test_truth_flows_reconstruct_state_and_oracles_do_not_mutate_dataset():
    rows = fixture_rows()
    before = copy.deepcopy(rows)
    outputs = diagnostic.rollout(Probe(),rows,0,'all_flows_oracle')
    assert {k:outputs[-1]['prediction'][k] for k in STATES}=={k:rows[0]['features'][k] for k in STATES}
    assert rows==before


def test_horizon_truth_must_equal_sum_of_disjoint_cohorts():
    rows = fixture_rows()
    rows[0]['labels']['5']['cohort_ttft_met'] += 1
    with pytest.raises(ValueError, match='numeric mismatch'):
        diagnostic.rollout(Probe(),rows,0,'free')


def test_history_groups_partition_only_original_nonstate_history():
    assert set(diagnostic.HISTORY)==set(diagnostic.ARRIVAL_HISTORY+diagnostic.ACTIVITY_HISTORY+diagnostic.OUTCOME_HISTORY)
    assert not set(diagnostic.HISTORY)&set(STATES)


def test_archived_coefficient_guard_rejects_material_change():
    with pytest.raises(ValueError, match='numeric mismatch'):
        diagnostic.close({'coef':[.1,.2]}, {'coef':[.1,.21]})
