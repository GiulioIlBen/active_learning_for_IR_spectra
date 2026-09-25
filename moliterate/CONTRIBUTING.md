## Contributing

### Testing

Add `moliterate/.env` file with:
```
AMSBASHRCPATH="path/to/amsbashrc.sh"
SCM_PYTHONDIR=".amspy"
```

Test recipes are present in the [justfile](./justfile) ([just-github](https://github.com/casey/just)).

Run all the tests:
```
just test
```

### Concepts
We fist look at the main folders:
```
└── moliterate
    ├── core                : contains the core components and the abstract classes that define the logic how objects interacts.
    ├── interfaces          : implements over the core classes BaseMolIterate and BaseMolDB
    ├── transforms          : provide simple transformation of the properties/metadata of MolDataRow (like naming, units, shape,...)
    ├── partition_criteria  : provide 1d, 2d or multi dimensional np.array of the data with 0 axis length corresponding to the len(db)
    └── filters             : based on partition_criteria and further operation we perform filtering of the BaseMolIterate

```

A fine-grained view of the folder structure is reported below. It is ordered and some files are disposed with the > stating "simple concept.py" > "complex concept depend on previous one" (like properties_info.py is a very simple class that is used by base_mol_iterate to provide property information about name/unit/shape: properties_info.py > base_mol_iterate.py)
```
moliterate
├── __init__.py
├── py.typed
├── core
│   ├── __init__.py
│   ├── interface.py
│   ├── mol_data_row.py
│   ├── properties_info.py > base_chem_dataset.py (+concat_chem_dataset.py) > base_chem_dataset_writer.py
│   ├── transforms.py 
│   └── partition_criterion.py > filters.py
├── interfaces
│   ├── ase_database.py
│   ├── in_memory.py
│   ├── params_data.py
│   └── rkf_files.py
├── transforms
│   ├── __init__.py
│   ├── atoms_stats.py
│   └── property_info_transform.py
|
└── partition_criteria   >>>>>>>>>>>>>>>── filters         
    ├── __init__.py                        ├── __init__.py     
    ├── criteria_1d.py                     ├── conditions_filter.py         
    ├── criteria_1d_splitting.py           ├── duplicates_rmsd.py                     
    ├── criteria_2d.py                     ├── multi_and_filter.py         
    └── criteria_3d.py                     └── scalar_filters.py         
```

#### Indexes: relative, absolute and origin

It is as following:
- idx_relative is the one obtained from enumerate, and what the user will mainly interact with.
- idx_absolute is the idx if you enumerate with subset_idxs = None
- idx_origin is the idx that you can use with native reader of that data source (e.g for a dictionary it will be the unique key for obtaining that entry, or some datasources have 1-based counting instead of the pythonic 0-based one)

```python
from scm.moliterate import load_dataset
# load iterable
ds = load_dataset(path)

# difference between idx_relative, idx_absolute, idx_origin
for idx_relative, entry in enumerate(ds):
    entry.idx_absolute
    entry.idx_origin
```

#### How to implement a iterable interface?

Look at the tutorial: [advanced-create-an-interface.ipynb](tutorials/advanced-create-an-interface.ipynb)

#### Integrations

```python
from typing import Union
from pydantic import BaseModel, field_validator
from scm.moliterate import load_dataset
from scm.moliterate.interfaces import ASEMolData, RKFMolData


class Workflow(BaseModel):
    data:Union[ASEMolData, RKFMolData]

    @field_validator("data", mode="before")
    @classmethod
    def _validate_metadata(cls, value):
        return load_dataset(**value)

path = ""
wk = Workflow(**{"data":{"data_source":path}})
print(wk.data)
```

### Related projects

Other great repos:

| Name                                                                                                  | Link DB-doc                                                                                   | Last commit                                                                                                                                                                            | Commit activity                                                                                                                                                                             | Commets                |
| ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| [best-of-atomistic-machine-learning](https://github.com/JuDFTteam/best-of-atomistic-machine-learning) | [link](https://github.com/JuDFTteam/best-of-atomistic-machine-learning#readme)                | [![Last commit](https://img.shields.io/github/last-commit/JuDFTteam/best-of-atomistic-machine-learning)](https://github.com/JuDFTteam/best-of-atomistic-machine-learning/commits/main) | [![Commit activity](https://img.shields.io/github/commit-activity/y/JuDFTteam/best-of-atomistic-machine-learning)](https://github.com/JuDFTteam/best-of-atomistic-machine-learning/commits) | Source for extra infos |
| [RDKit](https://github.com/rdkit/rdkit)                                                               | [link](https://www.rdkit.org/docs/Cartridge.html)                                             | [![Last commit on master](https://img.shields.io/github/last-commit/rdkit/rdkit/master)](https://github.com/rdkit/rdkit/commits/master)<br>                                            | [![Commit activity](https://img.shields.io/github/commit-activity/y/rdkit/rdkit)](https://github.com/rdkit/rdkit/commits)<br>                                                               |                        |
| [ASE](https://gitlab.com/ase/ase)                                                                     | [link](https://gitlab.com/ase/ase/-/tree/master/ase/db?ref_type=heads)                        |                                                                                                                                                                                        |                                                                                                                                                                                             |                        |
| [schnetpack](https://github.com/atomistic-machine-learning/schnetpack)                                | [link](https://schnetpack.readthedocs.io/en/latest/tutorials/tutorial_01_preparing_data.html) | [![Last commit](https://img.shields.io/github/last-commit/atomistic-machine-learning/schnetpack)](https://github.com/atomistic-machine-learning/schnetpack/commits/main)               | [![Commit activity](https://img.shields.io/github/commit-activity/y/atomistic-machine-learning/schnetpack)](https://github.com/atomistic-machine-learning/schnetpack/commits)               |                        |
| [load-atoms](https://github.com/jla-gardner/load-atoms)                                               | [link](https://github.com/jla-gardner/load-atoms#readme)                                      | [![Last commit](https://img.shields.io/github/last-commit/jla-gardner/load-atoms)](https://github.com/jla-gardner/load-atoms/commits/main)                                             | [![Commit activity](https://img.shields.io/github/commit-activity/y/jla-gardner/load-atoms)](https://github.com/jla-gardner/load-atoms/commits)                                             |                        |
| [OPTIMADE](https://github.com/Materials-Consortia/OPTIMADE)                                           | [link](https://optimade.org/optimade-python-tools/)                                           | [![Last commit](https://img.shields.io/github/last-commit/Materials-Consortia/OPTIMADE)](https://github.com/Materials-Consortia/OPTIMADE/commits/master)                               | [![Commit activity](https://img.shields.io/github/commit-activity/y/Materials-Consortia/OPTIMADE)](https://github.com/Materials-Consortia/OPTIMADE/commits)                                 |                        |
| [pymatgen-db](https://github.com/materialsproject/pymatgen-db)                                        | [link](https://materialsproject.github.io/pymatgen-db/)                                       | [![Last commit](https://img.shields.io/github/last-commit/materialsproject/pymatgen-db)](https://github.com/materialsproject/pymatgen-db/commits/master)                               | [![Commit activity](https://img.shields.io/github/commit-activity/y/materialsproject/pymatgen-db)](https://github.com/materialsproject/pymatgen-db/commits)                                 |                        |
| [moyo](https://github.com/spglib/moyo)                                                                | [link](https://spglib.github.io/moyo/python/)                                                 | [![Last commit on main](https://img.shields.io/github/last-commit/spglib/moyo/main)](https://github.com/spglib/moyo/commits/main)                                                      | [![Commit activity](https://img.shields.io/github/commit-activity/y/spglib/moyo)](https://github.com/spglib/moyo/commits)                                                                   |                        |