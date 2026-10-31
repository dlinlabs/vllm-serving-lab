import json

import pytest

from audit_interventions import audit_isolated, composite_good


def test_composite_score_does_not_hide_decode_slo_failure():
    row = dict(scheduled_at=10,first_content_at=10.1,terminal_at=11.2,
               usage={"completion_tokens":11})
    assert not composite_good(row)
    row["terminal_at"]=10.5
    assert composite_good(row)
    row["first_content_at"]=10.6
    assert not composite_good(row)
    row.update(first_content_at=10.1,terminal_at=10.1,usage={"completion_tokens":1})
    assert composite_good(row)


def test_isolated_audit_rejects_partial_or_failed_collection(tmp_path):
    path = tmp_path/"isolated.json"
    path.write_text(json.dumps({"completed":False}))
    with pytest.raises(ValueError,match="incomplete"):
        audit_isolated(tmp_path)
