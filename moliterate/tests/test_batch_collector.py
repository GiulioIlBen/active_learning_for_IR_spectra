import pytest
from ase import Atoms

from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.utils.batch_collector import batch_collector


def test_batch_collector_flushes_full_batches_only():
    flushed = []

    def flush(batch):
        flushed.append(list(batch))

    with batch_collector(flush=flush, batch_size=2) as add:
        add(1)
        add(2)
        add(3)
        add(4)

    assert flushed == [[1, 2], [3, 4]]


def test_batch_collector_flushes_remaining_items_on_exit():
    flushed = []

    def flush(batch):
        flushed.append(list(batch))

    with batch_collector(flush=flush, batch_size=2) as add:
        add(1)
        add(2)
        add(3)

    assert flushed == [[1, 2], [3]]


def test_batch_collector_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        with batch_collector(flush=lambda _: None, batch_size=0):
            pass


def test_in_memory_batch_collector_adds_rows():
    db = InMemoryMolData.create(available_properties=[PropertyInfo(name="energy")])
    rows = [
        ChemDataEntry(system=Atoms(), properties={"energy": 1.0, "other": 2.0}),
        ChemDataEntry(system=Atoms(), properties={"energy": 2.0, "other": 3.0}),
        ChemDataEntry(system=Atoms(), properties={"energy": 3.0, "other": 4.0}),
    ]

    with db.batch_collector(batch_size=2) as add:
        for row in rows:
            add(row)

    assert db.total_len() == 3
    assert [row.properties for row in db] == [{"energy": 1.0}, {"energy": 2.0}, {"energy": 3.0}]
