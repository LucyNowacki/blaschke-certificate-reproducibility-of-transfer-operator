"""
Map-generic high-precision Perron--Frobenius Galerkin--EDMD workers.

This module is generated for transfer_spectrum_lab.  It evaluates serialised
inverse-branch specifications of the form
    {"name":"formula_inverse_branches", "params": {"branches": [...]}}
and assembles the Legendre--Gauss transfer block.
"""
from __future__ import annotations

import copy
import math
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple

import mpmath as mp

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


WORKER_API_VERSION = "2026-08-02-process-row-block-assembly"


def _assembly_settings(kwargs):
    '''Resolve explicit or environment-controlled row-block assembly settings.'''
    raw_workers = kwargs.get("assembly_workers", None)
    if raw_workers is None:
        raw_workers = os.environ.get("MPMATH_PF_ASSEMBLY_WORKERS", "1")
    workers = max(1, int(raw_workers or 1))

    row_block_size = kwargs.get("row_block_size", None)
    if row_block_size is None:
        row_block_size = os.environ.get("MPMATH_PF_ROW_BLOCK_SIZE", None)
    if row_block_size in (None, "", 0, "0"):
        row_block_size = None
    else:
        row_block_size = max(1, int(row_block_size))
    return workers, row_block_size


def frobenius_norm_mp(A):
    '''Return the high-precision Frobenius norm of an mpmath matrix.'''
    s = mp.mpf("0")
    for i in range(A.rows):
        for j in range(A.cols):
            s += abs(A[i, j]) ** 2
    return mp.sqrt(s)


def gauss_legendre_mp(M: int):
    '''Return the M-point Gauss--Legendre nodes and weights as mpmath scalars.'''
    M = int(M)
    xs, ws = mp.gauss_quadrature(M, "legendre")
    return [mp.mpf(xs[j]) for j in range(M)], [mp.mpf(ws[j]) for j in range(M)]


def ortho_legendre_values(N: int, x):
    '''Evaluate the first N L2-normalised Legendre polynomials at x.'''
    N = int(N)
    if N <= 0:
        return []
    P = [mp.mpc(0) for _ in range(N)]
    P[0] = mp.mpc(1)
    if N >= 2:
        P[1] = mp.mpc(x)
        for n in range(1, N - 1):
            P[n + 1] = ((2*n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
    return [mp.sqrt(mp.mpf(2*n + 1) / 2) * P[n] for n in range(N)]


def _base_env():
    '''Build the restricted mpmath expression environment used by map formulae.'''
    env = {
        "mp": mp,
        "pi": mp.pi,
        "j": mp.j,
        "e": mp.e,
        "sqrt": mp.sqrt,
        "sin": mp.sin,
        "cos": mp.cos,
        "tan": mp.tan,
        "asin": mp.asin,
        "acos": mp.acos,
        "atan": mp.atan,
        "exp": mp.exp,
        "log": mp.log,
        "tanh": mp.tanh,
        "cosh": mp.cosh,
        "sinh": mp.sinh,
        "arg": mp.arg,
        "conj": mp.conj,
        "re": mp.re,
        "im": mp.im,
        "abs": abs,
        "min": min,
        "max": max,
    }
    return env


def _eval_parameter(value, params=None):
    '''Evaluate a serialised scalar parameter using the restricted mpmath environment.'''
    if isinstance(value, (int, float, complex, mp.mpf, mp.mpc)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    env = _base_env()
    if params:
        env.update(params)
    try:
        return mp.mpf(text)
    except Exception:
        pass
    try:
        return eval(text, {"__builtins__": {}}, env)
    except Exception:
        return text


_SIGNED_FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PY_COMPLEX_J_RE = re.compile(rf"^\(?\s*({_SIGNED_FLOAT_RE})\s*({_SIGNED_FLOAT_RE})j\s*\)?$")


def _mp_complex_from_value(value):
    '''Parse a real or complex scalar into an mpmath complex number.'''
    if isinstance(value, mp.mpc):
        return mp.mpc(value)
    if isinstance(value, mp.mpf):
        return mp.mpc(value)
    if isinstance(value, complex):
        return mp.mpc(str(value.real), str(value.imag))
    if isinstance(value, (int, float)):
        return mp.mpc(str(value))
    text = str(value).strip()
    try:
        return mp.mpc(text)
    except Exception:
        pass
    compact = text.replace(" ", "")
    match = _PY_COMPLEX_J_RE.fullmatch(compact)
    if match:
        return mp.mpc(mp.mpf(match.group(1)), mp.mpf(match.group(2)))
    if compact.startswith("(") and compact.endswith(")"):
        compact = compact[1:-1]
    if compact.endswith("j"):
        return mp.mpc(0, mp.mpf(compact[:-1] or "1"))
    return mp.mpc(_eval_parameter(text))


def _mp_complex_from_cluster_record(record):
    '''Read a full-precision target value from a serialised cluster record.'''
    # Prefer the full-precision serialisation.  The ``*_float`` fields exist
    # only for plotting and tabular convenience; using them for matching
    # silently imposes a binary64 floor on high-precision reference errors.
    for key in ("target", "value"):
        value = record.get(key)
        if value is not None and str(value).strip() not in {"", "nan", "None"}:
            return _mp_complex_from_value(value)

    real = record.get("target_real")
    if real is None:
        real = record.get("target_real_float")
    if real is None:
        real = record.get("real")
    imag = record.get("target_imag")
    if imag is None:
        imag = record.get("target_imag_float")
    if imag is None:
        imag = record.get("imag")
    if real is not None or imag is not None:
        return mp.mpc(mp.mpf(str(real if real is not None else 0)), mp.mpf(str(imag if imag is not None else 0)))
    return mp.mpc(0)


def _normalise_reference_clusters(reference_clusters):
    '''Convert serialised reference clusters into the internal cluster format.'''
    clusters = []
    for idx, cl in enumerate(reference_clusters):
        clusters.append({
            "name": str(cl.get("name", f"ref^{idx + 1}")),
            "family": str(cl.get("family", "reference")),
            "power": int(cl.get("power", idx + 1)),
            "multiplicity": int(cl.get("multiplicity", 1)),
            "value": _mp_complex_from_cluster_record(cl),
        })
    clusters.sort(key=lambda c: abs(c["value"]), reverse=True)
    return clusters


def _normalised_params(map_spec: Dict[str, Any]) -> Dict[str, Any]:
    '''Evaluate and return the parameter dictionary of a serialised map specification.'''
    raw = dict(map_spec.get("params", {}).get("parameters", {}))
    params: Dict[str, Any] = {}
    # Do two passes so parameters may refer to previous parameters.
    for _ in range(2):
        for k, v in raw.items():
            params[k] = _eval_parameter(v, params)
    return params


def _eval_expr(expr: str, params: Dict[str, Any], x=None):
    '''Evaluate a branch formula at x in the restricted mpmath environment.'''
    env = _base_env()
    env.update(params)
    if x is not None:
        env["x"] = x
    return eval(str(expr), {"__builtins__": {}}, env)


class FormulaTransferMap:
    def __init__(self, map_spec: Dict[str, Any]):
        '''Initialise a transfer map from serialised inverse-branch formulae.'''
        self.map_spec = serialisable_map_spec(map_spec)
        self.params_block = self.map_spec.get("params", {})
        self.label = str(self.params_block.get("label", self.map_spec.get("name", "map")))
        self.parameters = _normalised_params(self.map_spec)
        self.branch_specs = list(self.params_block.get("branches", []))

    def branches(self):
        '''Return the configured branch labels in their declared order.'''
        return [b.get("label", i + 1) for i, b in enumerate(self.branch_specs)]

    def _branch_spec(self, label):
        '''Resolve a branch label to its serialised formula specification.'''
        for b in self.branch_specs:
            if str(b.get("label")) == str(label):
                return b
        # fall back to one-based index
        try:
            return self.branch_specs[int(label) - 1]
        except Exception as exc:
            raise KeyError(f"unknown branch label {label!r}") from exc

    def tau(self, label, x):
        '''Evaluate the inverse branch tau for the given label and point x.'''
        b = self._branch_spec(label)
        return _eval_expr(b["tau"], self.parameters, x=x)

    def phi(self, label, x):
        '''Evaluate the transfer weight phi for the given label and point x.'''
        b = self._branch_spec(label)
        return _eval_expr(b["phi"], self.parameters, x=x)


def serialisable_map_spec(map_spec: Dict[str, Any] | None = None, **kwargs) -> Dict[str, Any]:
    '''Return an isolated deep copy of a map specification suitable for workers.'''
    if map_spec is None:
        map_spec = kwargs.get("map_spec", None)
    if map_spec is None:
        raise ValueError("serialisable_map_spec requires a map_spec dictionary")
    return copy.deepcopy(map_spec)


def make_transfer_map(map_spec: Dict[str, Any]):
    '''Construct the supported formula-based transfer-map evaluator.'''
    map_spec = serialisable_map_spec(map_spec)
    if map_spec.get("name") != "formula_inverse_branches":
        raise ValueError(f"unsupported map specification name {map_spec.get('name')!r}")
    return FormulaTransferMap(map_spec)


def exact_clusters_for_map(map_spec: Dict[str, Any], max_power: int = 12, include_trivial: bool = True, full_multiplicity: bool = True):
    '''Build the ordered exact spectral-target clusters declared by a map specification.'''
    map_spec = serialisable_map_spec(map_spec)
    params_block = map_spec.get("params", {})
    raw_clusters = list(params_block.get("exact_clusters", []))
    if not raw_clusters:
        return []
    params = _normalised_params(map_spec)
    clusters = []
    for cl in raw_clusters:
        power = int(cl.get("power", 0))
        if power > int(max_power):
            continue
        name = str(cl.get("name", f"target^{len(clusters)+1}"))
        if name == "1" and not include_trivial:
            continue
        value = _eval_parameter(cl.get("value"), params)
        mult = int(cl.get("multiplicity", 1))
        if not full_multiplicity:
            mult = 1
        clusters.append({
            "name": name,
            "value": value,
            "multiplicity": mult,
            "family": str(cl.get("family", "target")),
            "power": power,
        })
    clusters.sort(key=lambda c: abs(c["value"]), reverse=True)
    return clusters


def _assemble_transfer_row_block_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    '''Assemble one contiguous output-row block of the transfer and Gram matrices.'''
    mp.mp.dps = int(payload.get("dps", mp.mp.dps))
    N = int(payload["N"])
    M = int(payload["M"])
    k0 = int(payload["k0"])
    k1 = int(payload["k1"])
    return_gram = bool(payload.get("return_gram", True))
    fmap = make_transfer_map(payload["map_spec"])
    xs, ws = gauss_legendre_mp(M)
    B_rows = [[mp.mpc(0) for _ in range(N)] for _ in range(k1 - k0)]
    G_rows = [[mp.mpc(0) for _ in range(N)] for _ in range(k1 - k0)] if return_gram else None
    for j in range(M):
        xj = xs[j]
        wj = ws[j]
        p_x = ortho_legendre_values(N, xj)
        h_row = [mp.mpc(0) for _ in range(N)]
        for branch in fmap.branches():
            tau_val = fmap.tau(branch, xj)
            phi_val = fmap.phi(branch, xj)
            p_tau = ortho_legendre_values(N, tau_val)
            for ell in range(N):
                h_row[ell] += phi_val * p_tau[ell]
        for local_k, k in enumerate(range(k0, k1)):
            pk = mp.conj(p_x[k])
            for ell in range(N):
                B_rows[local_k][ell] += wj * pk * h_row[ell]
            if return_gram:
                for ell in range(N):
                    G_rows[local_k][ell] += wj * pk * p_x[ell]
    return {"k0": k0, "k1": k1, "B_rows": B_rows, "G_rows": G_rows}


def _assemble_transfer_block_parallel(N: int, M: int, map_spec: Dict[str, Any], return_gram: bool, dps: int, assembly_workers: int, row_block_size):
    '''Assemble a transfer block by distributing disjoint row blocks across processes.'''
    N = int(N)
    workers = max(1, int(assembly_workers or 1))
    if row_block_size is None:
        row_block_size = max(1, math.ceil(N / workers))
    row_block_size = max(1, int(row_block_size))
    tasks = [
        {
            "N": N,
            "M": int(M),
            "map_spec": map_spec,
            "return_gram": bool(return_gram),
            "dps": int(dps),
            "k0": k0,
            "k1": min(N, k0 + row_block_size),
        }
        for k0 in range(0, N, row_block_size)
    ]
    B = mp.matrix(N, N)
    Gram = mp.matrix(N, N) if return_gram else None
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_assemble_transfer_row_block_job, task) for task in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            k0 = int(result["k0"])
            for local_k, row in enumerate(result["B_rows"]):
                for ell, value in enumerate(row):
                    B[k0 + local_k, ell] = value
            if return_gram:
                for local_k, row in enumerate(result["G_rows"]):
                    for ell, value in enumerate(row):
                        Gram[k0 + local_k, ell] = value
    return (B, Gram) if return_gram else B


def assemble_transfer_block(N: int, M: int, map_spec: Dict[str, Any] | None = None, return_gram: bool = True, **kwargs):
    '''Assemble the raw N-by-N Legendre--Gauss transfer block and optional Gram block.'''
    N = int(N); M = int(M)
    dps = int(kwargs.get("dps", mp.mp.dps))
    mp.mp.dps = dps
    if map_spec is None:
        # Backwards-compatible Blaschke shorthand.
        mu = mp.mpf(str(kwargs.get("mu", kwargs.get("mu_str", "0.3"))))
        alpha = (1 + mu) / 2
        map_spec = {
            "name": "formula_inverse_branches",
            "params": {
                "label": "blaschke_mu_0p3",
                "parameters": {"mu": mp.nstr(mu, mp.mp.dps), "alpha": mp.nstr(alpha, mp.mp.dps)},
                "branches": [
                    {"label": 1, "tau": "x/2 - acos(mu*cos(pi*x/2))/pi", "phi": "mp.mpf('0.5') - mu/2*sin(pi*x/2)/sqrt(1 - mu**2*cos(pi*x/2)**2)"},
                    {"label": 2, "tau": "x/2 + acos(mu*cos(pi*x/2))/pi", "phi": "mp.mpf('0.5') + mu/2*sin(pi*x/2)/sqrt(1 - mu**2*cos(pi*x/2)**2)"},
                ],
            },
        }
    assembly_workers, row_block_size = _assembly_settings(kwargs)
    if assembly_workers > 1 and N > 1:
        return _assemble_transfer_block_parallel(
            N,
            M,
            map_spec,
            return_gram=return_gram,
            dps=dps,
            assembly_workers=assembly_workers,
            row_block_size=row_block_size,
        )

    fmap = make_transfer_map(map_spec)
    xs, ws = gauss_legendre_mp(M)
    B = mp.matrix(N, N)
    Gram = mp.matrix(N, N) if return_gram else None
    for j in range(M):
        xj = xs[j]
        wj = ws[j]
        p_x = ortho_legendre_values(N, xj)
        h_row = [mp.mpc(0) for _ in range(N)]
        for branch in fmap.branches():
            tau_val = fmap.tau(branch, xj)
            phi_val = fmap.phi(branch, xj)
            p_tau = ortho_legendre_values(N, tau_val)
            for ell in range(N):
                h_row[ell] += phi_val * p_tau[ell]
        for k in range(N):
            pk = mp.conj(p_x[k])
            for ell in range(N):
                B[k, ell] += wj * pk * h_row[ell]
            if return_gram:
                for ell in range(N):
                    Gram[k, ell] += wj * pk * p_x[ell]
    return (B, Gram) if return_gram else B


def assemble_pure_scaled_transfer_block(N: int, M: int, r, map_spec: Dict[str, Any], **kwargs):
    '''Conjugate the raw transfer block by the pure Legendre scaling diag(r**n).'''
    dps = kwargs.get("dps")
    if dps is not None:
        mp.mp.dps = int(dps)
    assembly_workers, row_block_size = _assembly_settings(kwargs)
    B = assemble_transfer_block(
        N,
        M,
        map_spec=map_spec,
        return_gram=False,
        dps=int(kwargs.get("dps", mp.mp.dps)),
        assembly_workers=assembly_workers,
        row_block_size=row_block_size,
    )
    r = mp.mpf(str(r))
    N = int(N)
    out = mp.matrix(N, N)
    scales = [r**n for n in range(N)]
    for i in range(N):
        for j in range(N):
            out[i, j] = scales[i] * B[i, j] / scales[j]
    return out


def _serialise_complex(z, digits=90):
    '''Serialise an mpmath complex value while suppressing negligible imaginary parts.'''
    z = mp.mpc(z)
    if abs(mp.im(z)) < mp.mpf(10) ** (-(min(int(digits), 80)//2)):
        return mp.nstr(mp.re(z), digits)
    return mp.nstr(z, digits)


def _float_or_nan(x):
    '''Convert x to a binary float, returning NaN when conversion is unavailable.'''
    try:
        return float(x)
    except Exception:
        return float('nan')


def greedy_cluster_errors(eigvals, clusters, max_clusters: int):
    '''Greedily match unused eigenvalues to target clusters and return raw cluster errors.'''
    pool = list(eigvals)
    rows = []
    for cl in clusters[:int(max_clusters)]:
        target = cl["value"]
        mult = int(cl.get("multiplicity", 1))
        if len(pool) < mult:
            rows.append({**cl, "target": target, "error": mp.nan, "selected": []})
            continue
        idxs = sorted(range(len(pool)), key=lambda i: abs(pool[i] - target))[:mult]
        selected = [pool[i] for i in idxs]
        err = max(abs(z - target) for z in selected)
        for i in sorted(idxs, reverse=True):
            del pool[i]
        rows.append({**cl, "target": target, "error": err, "selected": selected})
    return rows


def _rows_from_match(N, M, map_label, rows, gram_err, eigvals):
    '''Convert matched clusters and eigenvalues into serialisable tabular records.'''
    out = []
    for r in rows:
        target = r.get("target", r.get("value"))
        selected = r.get("selected", [])
        error = r.get("error", mp.nan)
        err_float = _float_or_nan(error) if error == error else float('nan')
        target_c = mp.mpc(target)
        out.append({
            "N": int(N),
            "M": int(M),
            "map_name": str(map_label),
            "name": str(r.get("name")),
            "family": str(r.get("family", "target")),
            "power": int(r.get("power", 0)),
            "multiplicity": int(r.get("multiplicity", 1)),
            "target": _serialise_complex(target),
            "target_float": _float_or_nan(mp.re(target_c)) if abs(mp.im(target_c)) == 0 else float('nan'),
            "target_real_float": _float_or_nan(mp.re(target_c)),
            "target_imag_float": _float_or_nan(mp.im(target_c)),
            "target_abs_float": _float_or_nan(abs(target_c)),
            "error": "nan" if not (error == error) else mp.nstr(error, 90),
            "error_float": err_float,
            "selected": [_serialise_complex(z) for z in selected],
            "gram_error": mp.nstr(gram_err, 90),
            "gram_error_float": _float_or_nan(gram_err),
        })
    eig_rows = []
    for k, z in enumerate(eigvals):
        zc = mp.mpc(z)
        eig_rows.append({
            "N": int(N),
            "M": int(M),
            "map_name": str(map_label),
            "eig_index": int(k),
            "eig": _serialise_complex(z),
            "eig_re": _float_or_nan(mp.re(zc)),
            "eig_im": _float_or_nan(mp.im(zc)),
            "eig_abs": _float_or_nan(abs(zc)),
            "gram_error": mp.nstr(gram_err, 90),
            "gram_error_float": _float_or_nan(gram_err),
        })
    return out, eig_rows


def _worker_one_pair(payload: Dict[str, Any]) -> Dict[str, Any]:
    '''Assemble, diagonalise, match, and serialise one requested pair (N, M).'''
    try:
        mp.mp.dps = int(payload.get("dps", 80))
        N = int(payload["N"]); M = int(payload["M"])
        map_spec = payload.get("map_spec")
        reference_clusters = payload.get("reference_clusters")
        B, Gram = assemble_transfer_block(
            N,
            M,
            map_spec=map_spec,
            return_gram=True,
            dps=int(payload.get("dps", mp.mp.dps)),
            assembly_workers=int(payload.get("assembly_workers", 1) or 1),
            row_block_size=payload.get("row_block_size", None),
        )
        eigvals = list(mp.eig(B, left=False, right=False))
        if reference_clusters:
            clusters = _normalise_reference_clusters(reference_clusters)
        else:
            clusters = exact_clusters_for_map(
                map_spec,
                max_power=int(payload.get("max_power", 12)),
                include_trivial=bool(payload.get("include_trivial", True)),
                full_multiplicity=bool(payload.get("full_multiplicity", True)),
            )
        rows = greedy_cluster_errors(eigvals, clusters, max_clusters=int(payload.get("max_clusters", len(clusters))))
        gram_err = frobenius_norm_mp(Gram - mp.eye(N))
        map_label = map_spec.get("params", {}).get("label", "map") if map_spec else "map"
        out_rows, eig_rows = _rows_from_match(N, M, map_label, rows, gram_err, eigvals)
        return {
            "ok": True,
            "N": N,
            "M": M,
            "rows": out_rows,
            "eig_rows": eig_rows,
            "gram_error": mp.nstr(gram_err, 90),
            "gram_error_float": _float_or_nan(gram_err),
            "assembly_mode": "process_row_blocks" if int(payload.get("assembly_workers", 1) or 1) > 1 else "serial",
            "assembly_workers": int(payload.get("assembly_workers", 1) or 1),
            "row_block_size": payload.get("row_block_size", None),
        }
    except Exception as e:
        return {"ok": False, "N": payload.get("N"), "M": payload.get("M"), "error": repr(e), "traceback": traceback.format_exc()}


def _run_tasks(tasks, workers=1, progress=True, desc="mpmath raw sweep"):
    '''Execute pair-worker tasks serially or in parallel and return sorted results.'''
    results = []
    workers = int(workers or 1)
    if workers <= 1:
        iterator = tasks
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc=desc, unit="job")
        for t in iterator:
            results.append(_worker_one_pair(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker_one_pair, t) for t in tasks]
            iterator = as_completed(futures)
            if progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc=desc, unit="job")
            for fut in iterator:
                results.append(fut.result())
    bad = [r for r in results if not r.get("ok")]
    if bad:
        raise RuntimeError("Some mpmath workers failed:\n" + "\n\n".join(str(b) for b in bad[:3]))
    results.sort(key=lambda r: (int(r["N"]), int(r["M"])))
    return results


def run_N_sweep_mpmath_raw(N_values: Iterable[int], map_spec: Dict[str, Any] | None = None, dps: int = 80, M_rule="square", workers: int = 1, max_power: int = 12, max_clusters: int = 11, include_trivial: bool = True, full_multiplicity: bool = True, progress: bool = True, reference_clusters=None, **kwargs):
    '''Run raw transfer spectra and cluster matching over an N-sweep with a chosen M rule.'''
    assembly_workers, row_block_size = _assembly_settings(kwargs)
    tasks = []
    for N in [int(v) for v in N_values]:
        M = N if M_rule == "square" else int(M_rule(N))
        tasks.append({
            "N": N, "M": M, "map_spec": map_spec, "dps": int(dps),
            "max_power": int(max_power), "max_clusters": int(max_clusters),
            "include_trivial": bool(include_trivial), "full_multiplicity": bool(full_multiplicity),
            "reference_clusters": reference_clusters,
            "assembly_workers": assembly_workers, "row_block_size": row_block_size,
        })
    if assembly_workers > 1:
        workers = 1
    results = _run_tasks(tasks, workers=workers, progress=progress, desc="strict mpmath raw N sweep")
    rows, eig_rows = [], []
    for res in results:
        rows.extend(res["rows"]); eig_rows.extend(res.get("eig_rows", []))
    return rows, {"jobs": results, "eig_rows": eig_rows}


def run_pair_sweep_mpmath_raw(pairs: Iterable[Tuple[int, int]], map_spec: Dict[str, Any] | None = None, dps: int = 80, workers: int = 1, max_power: int = 12, max_clusters: int = 11, include_trivial: bool = True, full_multiplicity: bool = True, progress: bool = True, reference_clusters=None, **kwargs):
    '''Run raw transfer spectra and cluster matching for explicitly supplied (N, M) pairs.'''
    assembly_workers, row_block_size = _assembly_settings(kwargs)
    tasks = []
    for N, M in pairs:
        tasks.append({
            "N": int(N), "M": int(M), "map_spec": map_spec, "dps": int(dps),
            "max_power": int(max_power), "max_clusters": int(max_clusters),
            "include_trivial": bool(include_trivial), "full_multiplicity": bool(full_multiplicity),
            "reference_clusters": reference_clusters,
            "assembly_workers": assembly_workers, "row_block_size": row_block_size,
        })
    if assembly_workers > 1:
        workers = 1
    results = _run_tasks(tasks, workers=workers, progress=progress, desc="strict mpmath raw pair sweep")
    rows, eig_rows = [], []
    for res in results:
        rows.extend(res["rows"]); eig_rows.extend(res.get("eig_rows", []))
    return rows, {"jobs": results, "eig_rows": eig_rows}


def build_reference_clusters_mpmath_raw(payloads: Iterable[Dict[str, Any]], dps: int = 80, workers: int = 1, assembly_workers: int = 1, row_block_size=None, progress: bool = True):
    '''Build numerical reference clusters from larger high-precision transfer blocks.'''
    out = []
    iterator = list(payloads)
    if progress and tqdm is not None:
        iterator = tqdm(iterator, desc="reference cluster builds", unit="map")
    for payload in iterator:
        mp.mp.dps = int(dps)
        map_spec = payload["map_spec"]
        N_ref = int(payload.get("N_ref", 60)); M_ref = int(payload.get("M_ref", N_ref + 10))
        max_targets = int(payload.get("max_targets", 25))
        skip_unit = bool(payload.get("skip_unit", True))
        B, Gram = assemble_transfer_block(
            N_ref,
            M_ref,
            map_spec=map_spec,
            return_gram=True,
            dps=int(dps),
            assembly_workers=int(assembly_workers or 1),
            row_block_size=row_block_size,
        )
        eigvals = list(mp.eig(B, left=False, right=False))
        eigvals.sort(key=lambda z: abs(z), reverse=True)
        refs = []
        serialisation_digits = max(30, int(dps))
        for z in eigvals:
            if skip_unit and abs(z - 1) < mp.mpf("1e-30"):
                continue
            if abs(z) < mp.mpf("1e-50"):
                continue
            refs.append({
                "name": f"ref^{len(refs)+1}",
                "value": _serialise_complex(z, serialisation_digits),
                "target": _serialise_complex(z, serialisation_digits),
                "multiplicity": 1,
                "family": "reference",
                "power": len(refs) + 1,
                "target_real_float": _float_or_nan(mp.re(z)),
                "target_imag_float": _float_or_nan(mp.im(z)),
                "target_abs_float": _float_or_nan(abs(z)),
            })
            if len(refs) >= max_targets:
                break
        gram_err = frobenius_norm_mp(Gram - mp.eye(N_ref))
        out.append({
            "map_label": payload.get("map_label", map_spec.get("params", {}).get("label", "map")),
            "N_ref": N_ref,
            "M_ref": M_ref,
            "reference_clusters": refs,
            "reference_gram_error": mp.nstr(gram_err, 90),
            "reference_gram_error_float": _float_or_nan(gram_err),
            "reference_mass_error": mp.nstr(gram_err, 90),
            "reference_mass_error_float": _float_or_nan(gram_err),
        })
    return out
