import pytest

from scm.moliterate import load_dataset
from scm.moliterate.interfaces import InMemoryMolData


def test_load_db_dump(db):
    print("========")
    if isinstance(db, InMemoryMolData):
        pytest.skip("InMemoryMolData cannot be loaded!")

    db0 = db.subset([1, 2])
    print(db0.model_dump())
    model_dict = db0.model_dump()
    db_ret = load_dataset(**model_dict)
    print(db_ret)
    assert db0.model_dump_json() == db_ret.model_dump_json()


def test_load_db_dump_concat(tuples_interface_path):
    pytest.importorskip("scm.plams")
    db = load_dataset(data_source=[p for _, p in tuples_interface_path[2:4]])
    model_dict = db.model_dump()
    db_ret = load_dataset(**model_dict)
    assert db.model_dump_json() == db_ret.model_dump_json()
