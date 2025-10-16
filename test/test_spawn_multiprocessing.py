import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _run_spawn_script(tmp_path, code):
    script = tmp_path / "spawn_check.py"
    script.write_text(textwrap.dedent(code))

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo_root / "src")

    subprocess.check_call([sys.executable, str(script)], env=env)


def test_spawn_end_to_end(tmp_path):
    """Exercise multiprocessing usage under the spawn start method."""

    code = """
    import multiprocessing as mp
    import numpy as np
    from quspin.basis import spin_basis_1d
    from quspin.operators import hamiltonian, quantum_operator
    from quspin.tools.block_tools import block_ops
    from quspin.tools import Floquet


    def drive(t):
        return np.cos(t)


    def pool_worker(payload):
        H, vec, time = payload
        return H.dot(vec, time=time)


    def quantum_worker(payload):
        op, vec, pars = payload
        return op.dot(vec, pars=pars)


    def main():
        mp.set_start_method("spawn", force=True)

        L = 4
        basis = spin_basis_1d(L, a=1)
        static = [["zz", [[1.0, i, (i + 1) % L] for i in range(L)]]]
        dynamic = [["x", [[0.5, i] for i in range(L)], drive, []]]
        H = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128)

        vec = np.ones(basis.Ns, dtype=np.complex128)
        vec /= np.linalg.norm(vec)

        ctx = mp.get_context("spawn")
        with ctx.Pool(2) as pool:
            pool.map(pool_worker, [(H, vec, 0.0), (H, vec, 0.5)])

        qop = quantum_operator({"zz": static}, basis=basis, dtype=np.complex128)
        with ctx.Pool(2) as pool:
            pool.map(quantum_worker, [(qop, vec, {"zz": 1.0})] * 2)

        blocks = [{"kblock": k, "a": 1} for k in range(L)]
        ops = block_ops(
            blocks,
            static,
            [],
            spin_basis_1d,
            (L,),
            np.complex128,
            basis_kwargs={"a": 1},
            save_previous_data=False,
        )

        times = np.linspace(0, 1, 3)
        for _ in ops.evolve(vec, 0.0, times, iterate=True, n_jobs=2, block_diag=False):
            pass
        ops.evolve(vec, 0.0, times, iterate=False, n_jobs=2, block_diag=False)

        Floquet.Floquet({"H": H, "T": 2 * np.pi}, UF=True, n_jobs=2)

        print("spawn workflow ok")


    if __name__ == "__main__":
        main()
    """

    _run_spawn_script(tmp_path, code)
