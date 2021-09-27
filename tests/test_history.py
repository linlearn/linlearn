from linlearn.solver import History
import pytest


def test_history():
    history = History("Truc", 4, verbose=False)
    history.allocate_record(1)  # param
    history.allocate_record(1)  # time
    history.update(0, n_iter=0, obj=1.2, tol=3.2, blabla=4, update_bar=False)
    history.update(0, n_iter=1, obj=1.2, tol=3.2, blabla=4)
    history.update(0, n_iter=2, obj=1.2, tol=3.2, blabla=4)
    history.update(0, n_iter=3, obj=1.2, tol=3.2, blabla=4)
    history.update(0, n_iter=4, obj=1.2, tol=3.2, blabla=4)
    with pytest.raises(ValueError) as exc_info:
        history.update(0, n_iter=5, obj=1.2, tol=3.2, blabla=4)
    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "Already 5 updates while max_iter=4"

    history.clear()
    history.update(0, n_iter=0, obj=1.2, tol=3.2, blabla=4, update_bar=False)
    history.update(0, n_iter=1, obj=1.2, tol=3.2, blabla=4)
    history.update(0, n_iter=2, obj=1.2, tol=3.2, blabla=4)
    history.update(0, n_iter=3, obj=1.2, tol=3.2, blabla=4)
    with pytest.raises(ValueError) as exc_info:
        history.update(0, n_iter=4, obj=1.2, blabla=4)
    assert exc_info.type is ValueError
    assert "'update' excepted the following keys:" in exc_info.value.args[0]

    # TODO: figure out how to test print()
