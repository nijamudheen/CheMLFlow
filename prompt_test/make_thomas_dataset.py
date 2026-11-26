#!/usr/bin/env python3
"""
Generate a Thomas cyclically-symmetric-attractor forecasting dataset, following the same
protocol as make_lv_dataset.py (one continuous trajectory, chronological split, fine RK4
integration, then downsampled).

    dx/dt = sin(y) - b*x
    dy/dt = sin(z) - b*y          (only x is observed, as a single variable)
    dz/dt = sin(x) - b*z

Thomas (1999). The system is cyclically symmetric in (x,y,z) and has a SINGLE parameter b.
It is dissipative (div f = -3b) and chaotic for b below ~0.208; b = 0.1 is the standard
chaotic setting, with a published largest Lyapunov exponent of roughly 0.05-0.08 per time
unit. This script MEASURES that exponent from the vector field (see below) rather than
asserting it, so the output can be certified as chaotic.

Protocol (mirrors make_lv_dataset.py):
  * One continuous series of N_TOTAL = 10,000 points, sampled at a fixed interval.
  * BURN_IN samples are integrated and discarded first, so the recorded series lies on the
    attractor rather than on the transient approaching it. (Lotka-Volterra needs no burn-in
    because every orbit is already periodic; a dissipative chaotic system does.)
  * The fine RK4 step is the sample interval divided by an integer number of substeps.
  * The first N_TOTAL - N_HIDDEN rows go to the agent; the last N_HIDDEN rows (the test
    block) are withheld. Inside the visible 9,000 the agent can reproduce the paper's
    500 / 7500 / 1000 warm-up / train / val layout itself.
  * Optional observation noise (paper Eq. 8): x_obs = x_clean + sigma * std(x_clean) * N(0,1),
    added to the WHOLE series (visible and hidden). Default sigma = 0 (no noise).
    The hidden file always keeps the clean series, so the evaluator scores against clean truth.
  * No normalization is applied: data are left in raw units on purpose.

Integrity checks written to the .meta.json:
  * lyapunov_exponent_per_time_unit - largest Lyapunov exponent by Benettin's algorithm
    (one tangent vector propagated through the analytic Jacobian, renormalized periodically).
    Positive => chaotic. Compare against the literature value for the chosen b.
  * lyapunov_time_samples - 1/lambda converted into sample units; this is what makes a
    forecast-horizon grid meaningful, since a horizon only means something relative to it.
  * decorrelation_samples - first zero crossing of the autocorrelation, the analogue of
    "samples per cycle" for a system that has no period.
  * rk4_halving_max_abs_diff - integration error proxy: the same initial condition re-run
    with twice as many RK4 substeps, compared over a SHORT window. Over long windows any
    two numerical solutions of a chaotic system separate exponentially by construction, so
    this is deliberately measured over --halving-window samples only.

Outputs
  --out-visible  agent file  : float64 .npy, shape (N_visible, 1), time on rows
  --out-hidden   hidden file : .npz with observed_full, clean_full, y_clean_full, z_clean_full, n_visible
  <hidden>.meta.json         : all generation parameters, measured exponent, checksums

Give agents ONLY the visible file. Keep the hidden file and its .meta.json out of their reach.
"""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np


def f(s, b):
    x, y, z = s
    return np.array([math.sin(y) - b * x,
                     math.sin(z) - b * y,
                     math.sin(x) - b * z])


def jac(s, b):
    x, y, z = s
    return np.array([[-b, math.cos(y), 0.0],
                     [0.0, -b, math.cos(z)],
                     [math.cos(x), 0.0, -b]])


def rk4_step(s, dt, b):
    k1 = f(s, b)
    k2 = f(s + .5 * dt * k1, b)
    k3 = f(s + .5 * dt * k2, b)
    k4 = f(s + dt * k3, b)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_step_tangent(s, w, dt, b):
    """Advance the state and one tangent vector together (variational equation w' = J(s) w)."""
    k1, l1 = f(s, b), jac(s, b) @ w
    k2, l2 = f(s + .5 * dt * k1, b), jac(s + .5 * dt * k1, b) @ (w + .5 * dt * l1)
    k3, l3 = f(s + .5 * dt * k2, b), jac(s + .5 * dt * k2, b) @ (w + .5 * dt * l2)
    k4, l4 = f(s + dt * k3, b), jac(s + dt * k3, b) @ (w + dt * l3)
    return (s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4),
            w + dt / 6 * (l1 + 2 * l2 + 2 * l3 + l4))


def lyapunov_benettin(s0, b, dt, n_steps, renorm_every=10, discard_frac=0.1):
    """Largest Lyapunov exponent (per time unit) from the analytic Jacobian.

    Benettin et al. (1980): propagate one tangent vector, renormalize to unit length every
    `renorm_every` steps, and accumulate the logarithms of the growth factors.
    """
    s = np.asarray(s0, float)
    w = np.array([1.0, 0.0, 0.0])
    w /= np.linalg.norm(w)
    acc, t_acc, logs = 0.0, 0.0, []
    n_discard = int(n_steps * discard_frac)
    for i in range(n_steps):
        for _ in range(renorm_every):
            s, w = rk4_step_tangent(s, w, dt, b)
        g = np.linalg.norm(w)
        w /= g
        if i >= n_discard:                 # let the tangent vector align with the dominant direction
            acc += math.log(g)
            t_acc += renorm_every * dt
            logs.append(acc / t_acc)
    return float(acc / t_acc), logs


def decorrelation_samples(x, maxlag=2000):
    """First zero crossing of the autocorrelation, in samples."""
    xc = x - x.mean()
    a = np.correlate(xc, xc, "full")[len(xc) - 1:len(xc) - 1 + maxlag]
    a /= a[0]
    idx = np.where(a <= 0)[0]
    return int(idx[0]) if len(idx) else None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-visible", required=True)
    p.add_argument("--out-hidden", required=True)
    p.add_argument("--n-total", type=int, default=10000)
    p.add_argument("--n-hidden", type=int, default=1000)
    p.add_argument("--burn-in", type=int, default=2000, help="samples integrated and discarded before recording")
    p.add_argument("--sample-dt", type=float, default=0.25, help="sample interval in time units")
    p.add_argument("--max-substep", type=float, default=0.01, help="largest RK4 step; substeps are an integer split of the sample interval")
    p.add_argument("--b", type=float, default=0.1, help="the system's only parameter; chaotic below ~0.208")
    p.add_argument("--x0", type=float, default=1.1)
    p.add_argument("--y0", type=float, default=1.1)
    p.add_argument("--z0", type=float, default=-0.01)
    p.add_argument("--noise-sigma", type=float, default=0.0, help="paper Eq.8 noise level (0, 0.1, 0.2, 0.3); 0 = none")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--lyap-steps", type=int, default=20000, help="renormalization intervals for the Benettin estimate")
    p.add_argument("--halving-window", type=int, default=200, help="samples over which the RK4 step-halving check is compared")
    a = p.parse_args()

    b = a.b
    s0 = np.array([a.x0, a.y0, a.z0], float)
    dt_s = a.sample_dt
    m = int(math.ceil(dt_s / a.max_substep))        # RK4 substeps per sample
    dt = dt_s / m

    # --- burn-in: reach the attractor before recording
    s = s0.copy()
    for _ in range(a.burn_in):
        for _ in range(m):
            s = rk4_step(s, dt, b)
    s_attractor = s.copy()

    # --- integrate and record; sample 0 is the first post-burn-in point
    n = a.n_total
    traj = np.empty((n, 3))
    for i in range(n):
        traj[i] = s
        for _ in range(m):
            s = rk4_step(s, dt, b)
    xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]

    if not np.all(np.isfinite(traj)):
        raise RuntimeError("trajectory diverged; check parameters")

    # --- integration-error proxy: same start, twice the substeps, compared over a short window
    s_h = s_attractor.copy()
    xh = np.empty(a.halving_window)
    for i in range(a.halving_window):
        xh[i] = s_h[0]
        for _ in range(2 * m):
            s_h = rk4_step(s_h, dt / 2, b)
    halving_diff = float(np.max(np.abs(xh - xs[:a.halving_window])))

    # --- certify chaos from the vector field
    lam, _ = lyapunov_benettin(s_attractor, b, dt, a.lyap_steps)
    if lam <= 0:
        print(f"WARNING: measured largest Lyapunov exponent {lam:.5f} <= 0; this b is not chaotic")
    lyap_time_tu = (1.0 / lam) if lam > 0 else None
    lyap_time_samples = (lyap_time_tu / dt_s) if lyap_time_tu else None
    decorr = decorrelation_samples(xs)

    # --- observation noise (paper Eq. 8), added to the whole series; clean copy kept for scoring
    clean = xs.copy()
    if a.noise_sigma > 0:
        rng = np.random.default_rng(a.noise_seed)
        observed = clean + a.noise_sigma * clean.std() * rng.standard_normal(n)
    else:
        observed = clean.copy()

    n_vis = n - a.n_hidden
    Path(a.out_visible).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_hidden).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out_visible, observed[:n_vis].reshape(-1, 1))
    np.savez(a.out_hidden, observed_full=observed, clean_full=clean,
             y_clean_full=ys, z_clean_full=zs, n_visible=np.array(n_vis))

    meta = {
        "system": "thomas_cyclically_symmetric", "observed_variable": "x",
        "equations": ["dx/dt = sin(y) - b*x", "dy/dt = sin(z) - b*y", "dz/dt = sin(x) - b*z"],
        "reference": "R. Thomas, Int. J. Bifurcation Chaos 9(10), 1889-1905 (1999)",
        "params": {"b": b, "x0": a.x0, "y0": a.y0, "z0": a.z0},
        "n_total": n, "n_visible": n_vis, "n_hidden": a.n_hidden, "burn_in_samples": a.burn_in,
        "paper_layout_inside_visible": {"warmup": 500, "train": 7500, "val": 1000, "test_hidden": a.n_hidden},
        "sample_dt": dt_s, "rk4_substeps_per_sample": m, "rk4_dt": dt,
        "divergence_of_flow": -3 * b,
        "lyapunov_exponent_per_time_unit": lam,
        "lyapunov_time_time_units": lyap_time_tu,
        "lyapunov_time_samples": lyap_time_samples,
        "lyapunov_method": f"Benettin, analytic Jacobian, {a.lyap_steps} renormalizations, first 10% discarded",
        "decorrelation_samples": decorr,
        "rk4_halving_max_abs_diff": halving_diff,
        "rk4_halving_window_samples": a.halving_window,
        "noise_sigma": a.noise_sigma, "noise_seed": a.noise_seed if a.noise_sigma > 0 else None,
        "normalized": False,
        "x_clean_std_visible": float(clean[:n_vis].std()),
        "x_clean_range_visible": [float(clean[:n_vis].min()), float(clean[:n_vis].max())],
        "x_clean_range_hidden": [float(clean[n_vis:].min()), float(clean[n_vis:].max())],
        "sha256_visible": sha256(a.out_visible), "sha256_hidden": sha256(a.out_hidden),
    }
    meta_path = Path(a.out_hidden).with_suffix(".meta.json")
    json.dump(meta, open(meta_path, "w"), indent=2)

    print(f"b={b} | sample interval {dt_s} | {m} RK4 substeps (dt={dt:.5f}) | burn-in {a.burn_in} samples")
    print(f"largest Lyapunov exponent {lam:.5f} / time unit  =>  Lyapunov time {lyap_time_tu:.2f} tu = {lyap_time_samples:.1f} samples")
    print(f"decorrelation (first ACF zero) {decorr} samples | flow divergence {-3*b}")
    print(f"RK4 step-halving max |diff| over {a.halving_window} samples: {halving_diff:.2e}")
    print(f"{n} points = {n_vis} visible + {a.n_hidden} hidden | noise sigma={a.noise_sigma}")
    print(f"x range visible {meta['x_clean_range_visible']}, hidden {meta['x_clean_range_hidden']}")
    print(f"agent file : {a.out_visible}  shape {(n_vis, 1)}\nhidden file: {a.out_hidden}\nmeta       : {meta_path}")


if __name__ == "__main__":
    main()
