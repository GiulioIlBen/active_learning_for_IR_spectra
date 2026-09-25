import pytest
from pydantic import ValidationError

from scm.moliterate.utils import progress_bar
from scm.moliterate.utils.hashable_model import FrozenHashableModel, hash_set, stable_json_digest


class ExampleModel(FrozenHashableModel):
    x: int
    y: int

    def hash_payload(self):
        return {"x": self.x, "y": self.y}


def test_stable_json_digest_is_order_independent():
    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}
    assert stable_json_digest(left) == stable_json_digest(right)


def test_hash_set_uses_model_content_digest():
    item = ExampleModel(x=1, y=2)
    digest = hash_set([item])
    expected = stable_json_digest([item._content_digest()])
    assert digest == expected


def test_hash_set_is_order_independent_for_values():
    assert hash_set([1, 2, 3]) == hash_set([3, 2, 1])


def test_frozen_hashable_model_is_immutable():
    item = ExampleModel(x=1, y=2)
    with pytest.raises(ValidationError):
        item.x = 3


def test_moliterate_tqdm_uses_len_and_config(monkeypatch):
    calls = {}

    def fake_tqdm(iterable, **kwargs):
        calls["iterable"] = iterable
        calls["kwargs"] = kwargs
        return ("tqdm", iterable, kwargs)

    monkeypatch.setattr(progress_bar, "_tqdm", fake_tqdm)
    monkeypatch.setattr(progress_bar, "MOLITERATE_PROGRESS_BAR_CONFIG", progress_bar.TqdmConfig(disable=True))

    result = progress_bar.moliterate_tqdm([1, 2, 3])

    assert result[0] == "tqdm"
    assert calls["kwargs"]["total"] == 3
    assert calls["kwargs"]["disable"] is True


def test_moliterate_tqdm_kwargs_override_defaults(monkeypatch):
    calls = {}

    def fake_tqdm(iterable, **kwargs):
        calls["iterable"] = iterable
        calls["kwargs"] = kwargs
        return ("tqdm", iterable, kwargs)

    monkeypatch.setattr(progress_bar, "_tqdm", fake_tqdm)
    monkeypatch.setattr(progress_bar, "MOLITERATE_PROGRESS_BAR_CONFIG", progress_bar.TqdmConfig(disable=False))

    progress_bar.moliterate_tqdm([1, 2], disable=True, total=10, desc="test")

    assert calls["kwargs"]["total"] == 10
    assert calls["kwargs"]["disable"] is True
    assert calls["kwargs"]["desc"] == "test"
