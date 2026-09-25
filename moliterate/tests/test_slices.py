from scm.moliterate.interfaces.in_memory import InMemoryMolData


def test_slices(db_in_memory: InMemoryMolData):
    sliced_im: InMemoryMolData = db_in_memory[[0, 1]]
    assert len(sliced_im) == 2
    sliced_im = db_in_memory[1:]
    assert len(sliced_im) == 2
    assert sliced_im[0].idx_origin == 1
