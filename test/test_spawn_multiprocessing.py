import multiprocessing as mp

import numpy as np
import pytest

from quspin.basis import spin_basis_1d, spin_basis_general
from quspin.operators import hamiltonian, quantum_operator


L = 4
_SPIN_BASIS = spin_basis_1d(L, Nup=2)


def _drive(time):
    return np.cos(time)


_BONDS = [[1.0, i, (i + 1) % L] for i in range(L)]
_H_SPIN = hamiltonian(
    [["zz", _BONDS]],
    [["xx", _BONDS, _drive, []]],
    basis=_SPIN_BASIS,
    dtype=np.float64,
    check_herm=False,
    check_symm=False,
    check_pcon=False,
)


_QUANTUM_OPERATOR = quantum_operator(
    {"J": [["zz", _BONDS]]},
    basis=_SPIN_BASIS,
    dtype=np.float64,
    check_herm=False,
    check_symm=False,
    check_pcon=False,
)

_QUANTUM_PARS = {"J": 1.5}


_MAP = np.roll(np.arange(L), -1)
_GENERAL_BASIS = spin_basis_general(L, Nup=L // 2, kblock=(_MAP, 0))
_H_GENERAL = hamiltonian(
    [["zz", _BONDS]],
    [],
    basis=_GENERAL_BASIS,
    dtype=np.float64,
    check_herm=False,
    check_symm=False,
    check_pcon=False,
)


def _hamiltonian_worker(args):
    vec, time = args
    return _H_SPIN.dot(vec, time=time)


def _quantum_operator_worker(vec):
    return _QUANTUM_OPERATOR.tohamiltonian(_QUANTUM_PARS).dot(vec)


def _general_basis_worker(vec):
    return _H_GENERAL.dot(vec)


@pytest.mark.skipif("spawn" not in mp.get_all_start_methods(), reason="spawn start method not available")
def test_hamiltonian_spawn_dot():
    payload = [
        (np.arange(_H_SPIN.Ns, dtype=np.float64), 0.25),
        (np.ones(_H_SPIN.Ns, dtype=np.float64), 1.75),
    ]
    for item in payload:
        _hamiltonian_worker(item)

    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        results = pool.map(_hamiltonian_worker, payload)

    assert len(results) == len(payload)
    for vec in results:
        assert vec.shape == (_H_SPIN.Ns,)


@pytest.mark.skipif("spawn" not in mp.get_all_start_methods(), reason="spawn start method not available")
def test_quantum_operator_spawn_dot():
    sample = np.linspace(0.0, 1.0, _QUANTUM_OPERATOR.Ns, dtype=np.float64)
    _quantum_operator_worker(sample)

    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        results = pool.map(_quantum_operator_worker, [sample, sample])

    assert len(results) == 2
    for vec in results:
        assert vec.shape == (_QUANTUM_OPERATOR.Ns,)


@pytest.mark.skipif("spawn" not in mp.get_all_start_methods(), reason="spawn start method not available")
def test_general_basis_spawn_dot():
    vector = np.arange(_H_GENERAL.Ns, dtype=np.float64)
    _general_basis_worker(vector)

    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        results = pool.map(_general_basis_worker, [vector, vector])

    assert len(results) == 2
    for vec in results:
        assert vec.shape == (_H_GENERAL.Ns,)
