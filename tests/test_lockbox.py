import json
import pytest

from state_conditioned_entropy_pooling import open_lockbox


def test_lockbox_refuses_new_design(tmp_path):
    path = tmp_path / "stamp.json"
    open_lockbox({"arm": "A"}, {"result": 1}, path)
    with pytest.raises(RuntimeError):
        open_lockbox({"arm": "B"}, {"result": 2}, path)
    payload = json.loads(path.read_text())
    assert payload["design"]["arm"] == "A"
