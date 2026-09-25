import pytest

from scm.moliterate.core.base_chem_dataset import IterDataCoreError
from scm.moliterate.core.interface import (
    ChemDataSetFormat,
    create_dataset,
    load_dataset,
    resolve_format,
)


def test_create_dataset_invalid_format_raises():
    with pytest.raises(IterDataCoreError, match="Cannot create dataset for format: not-a-format"):
        create_dataset(fmt="not-a-format")


def test_load_dataset_format_list_length_mismatch_raises():
    with pytest.raises(IterDataCoreError, match="format list must match in length"):
        load_dataset(data_source=["one.db", "two.db"], fmt=[ChemDataSetFormat.ASE])


def test_load_dataset_list_format_for_single_source_raises():
    with pytest.raises(IterDataCoreError, match="not valid for one data_source"):
        load_dataset(data_source="one.db", fmt=[ChemDataSetFormat.ASE])


def test_resolve_format_dir_defaults_params(tmp_path):
    data_source, fmt = resolve_format(str(tmp_path), fmt=None)
    assert data_source == str(tmp_path)
    assert fmt is ChemDataSetFormat.ParAMS


def test_resolve_format_appends_suffix_when_missing(tmp_path):
    data_source, fmt = resolve_format(str(tmp_path / "dataset"), fmt=ChemDataSetFormat.ASE)
    assert data_source.endswith(".db")
    assert fmt is ChemDataSetFormat.ASE


def test_resolve_format_missing_suffix_without_format_raises(tmp_path):
    with pytest.raises(IterDataCoreError, match="format is not given"):
        resolve_format(str(tmp_path / "dataset"), fmt=None)


def test_resolve_format_unsupported_extension_raises():
    with pytest.raises(IterDataCoreError, match="Unsupported file extension"):
        resolve_format("dataset.txt", fmt=None)
