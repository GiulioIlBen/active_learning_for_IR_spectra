from itertools import islice

from ase import Atoms


def test_simple_iter_operations(db):
    """run pytest with -s for visual inspection"""
    # test total_len
    length = len(db)
    # test str
    print(db)
    # test iter
    assert length == len(list(db))
    # test get_row
    row = db.get_row(0)
    assert row.idx_absolute == 0
    # test properties_md_table and available_properties
    av_p = db.available_properties
    print(av_p)
    table = db.properties_md_table()
    print(table)
    # test metadata
    print(db.metadata)


def test_iter_db_base(db):
    ROW_SAMPLE_SIZE = 5
    expected_rows = min(len(db), ROW_SAMPLE_SIZE)
    rows = list(islice(db, expected_rows))

    assert len(rows) == expected_rows
    assert [row.idx_absolute for row in rows] == list(range(expected_rows))
    assert all(isinstance(row.system, Atoms) for row in rows)

    for row in rows:
        assert isinstance(row.idx_absolute, int)
        fetched = db.get_row(row.idx_absolute)
        assert fetched.idx_absolute == row.idx_absolute
        assert fetched.idx_origin == row.idx_origin
        assert fetched.properties.keys() == row.properties.keys()
