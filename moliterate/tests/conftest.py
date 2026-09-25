from pathlib import Path

import pytest
from ase import Atoms

from scm.moliterate.core import ChemDataEntry, PropertyInfo
from scm.moliterate.interfaces.ase_database import ASEMolData
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.interfaces.params_data import ParAMSData
from scm.moliterate.interfaces.rkf_files import RKFMolData


@pytest.fixture
def db_in_memory():
    props_info = [PropertyInfo(name="energy")]
    db = InMemoryMolData.create(available_properties=props_info)
    rows = [
        ChemDataEntry(system=Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 1.5]]), properties={"energy": 1.0}),
        ChemDataEntry(system=Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 2.5]]), properties={"energy": 2.0}),
        ChemDataEntry(system=Atoms(numbers=[2], positions=[[0, 0, 0]]), properties={"energy": 1.5}),
    ]
    db.add_systems(rows)
    return db


TEST_FOLDER = Path(__file__).parent.absolute()

ITER_PATHS_AVAIL = [
    (InMemoryMolData, None),
    (ASEMolData, None),
    (RKFMolData, TEST_FOLDER.parent / "tutorials/data/molecule.rkf"),
    (RKFMolData, TEST_FOLDER.parent / "tutorials/data/periodic.rkf"),
    (ParAMSData, TEST_FOLDER.parent / "tutorials/data/params"),
    # (RKFMolData, TEST_FOLDER / "rkf/md/ams.rkf"),
    # (RKFMolData, TEST_FOLDER / "rkf/go/ams.rkf"),
]


def get_plams_tests():
    try:
        import scm.plams as plams
    except ImportError:
        return []
    RKF_PATHS = Path(plams.__path__[0]) / "unit_tests/rkf"
    if not RKF_PATHS.exists():
        RKF_PATHS = TEST_FOLDER.parent / "PLAMS" / "unit_tests/rkf"
        if not RKF_PATHS.exists():
            return []
    return [
        (RKFMolData, RKF_PATHS / "conformers/conformers.rkf"),
        (RKFMolData, RKF_PATHS / "propanenitrile/ams.rkf"),
        (RKFMolData, RKF_PATHS / "water_optimization/ams.rkf"),
        (RKFMolData, RKF_PATHS / "tools_plot/md/ams.rkf"),
    ]


ITER_PATHS_AVAIL.extend(get_plams_tests())


@pytest.fixture(scope="session")
def tuples_interface_path():
    return ITER_PATHS_AVAIL


@pytest.fixture(params=ITER_PATHS_AVAIL, scope="session")
def db_interface_tupl(request):
    db_interface_cls, data_source = request.param
    if db_interface_cls is ParAMSData:
        try:
            from scm.params import DataSet, JCEntry, ParAMSJob, ParAMSResults, ResultsImporter  # noqa: F401
            from scm.plams import Settings  # noqa: F401
        except ImportError:
            pytest.skip("ParAMS interface dependencies are not installed")
    if db_interface_cls is RKFMolData:
        try:
            from scm.plams import KFFile, KFHistory, Trajectory  # noqa: F401
        except ImportError:
            pytest.skip("RKF interface dependencies are not installed")
    return db_interface_cls, data_source


@pytest.fixture(scope="session")
def db(db_interface_tupl, tmp_path_factory):
    db_interface_cls = db_interface_tupl[0]
    data_source = db_interface_tupl[1]
    print(data_source)
    if data_source is None:
        temp_dir = tmp_path_factory.mktemp("moliterate_db")
        data_source = temp_dir / f"{db_interface_cls.__name__}.db"
        db = db_interface_cls.create(
            data_source=str(data_source), available_properties=[PropertyInfo(name="energy", unit="eV")]
        )
        db.add_systems(
            [
                ChemDataEntry(system=Atoms(), properties={"energy": -1.0}, metadata={"tag": "a"}),
                ChemDataEntry(system=Atoms(), properties={"energy": -2.0}, metadata={"tag": "b"}),
                ChemDataEntry(system=Atoms(), properties={"energy": -3.0}, metadata={"tag": "c"}),
            ]
        )
    else:
        db = db_interface_cls(data_source=str(data_source))
    yield db
    try:
        db.unlink(missing_ok=True)
    except Exception:
        pass
