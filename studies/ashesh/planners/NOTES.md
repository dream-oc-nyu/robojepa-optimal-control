# Gradient planners for reach-v0: `adam`, `trgn`

Contract: `bench/interface.py`. Plan `u ∈ R^{H×3}` (metres), features `z(u) = ctx.rollout(u)`, residual `r = z − g`,
`C(z) = mean|z − g|`, horizon weights `ω_h = γ^{H−1−h}` (γ = 0.99; final step weight 1), `L(u) = Σ_h ω_h C(z_h(u))`.

## Maths
**adam** (`adam.py`). Projected Adam on `L`: `u ← Proj(u − lr · m̂/(√v̂ + ε))`, `Proj` = row-wise clip to `‖u_h‖ ≤ step_bound_m`.
Gradient by reverse mode, or batched central differences (one rollout of 6H+1 plans). Starts from the warm start, else
zeros + seeded N(0, (1 mm)²). Runs to the budget, returns the best evaluated iterate. Uses no feedback.

**trgn** (`trgn.py`). Pseudo-Huber `φ(u) = Σ ω_h(√(r² + δ²) − δ)`, `δ = 0.1·median|r|`; IRLS weights `w = ω_h/√(r² + δ²)`.
With `J = ∂z/∂u ∈ R^{HP×3H}`, each inner iteration solves `(JᵀWJ + λI)d = −JᵀWr` with the smallest `λ ≥ λ_floor`
giving `‖d‖ ≤ Δ_in` (eigen-decomposition of the 3H×3H matrix, bisection on `‖d(λ)‖`), projects `u + d` onto
`{‖u_0‖ ≤ Δ, ‖u_h‖ ≤ step_bound_m}`, and accepts only if `L` decreases; `Δ_in` halves on rejection, doubles when the
linearised-L1 ratio > 0.75 at the boundary. The outer radius `Δ` (cap on the executed action) persists across ticks:
`ρ` None or < 0.25 → `Δ/2`; `ρ` > 0.75 and `‖u_0‖ ≥ 0.9Δ` → `2Δ`; else keep; `Δ ∈ [5 mm, step_bound_m]`, initial 5 cm.
`J` comes from one batched finite-difference rollout (`fd`, default, central, 1 mm) or 3H `torch.func.jvp` passes (`jvp`,
falls back to `fd` on any exception). Central differences at `u = 0` return only the odd, direction-dependent part of
the response: a term depending on `‖u‖` alone cancels (tested: `J_fd(0) = G·J_true` on the defective toy).

## Ablation points (all config fields)
- `trgn`: `max_inner_iters` (1 = one GN step from zero, uses only the odd response; default 50 re-linearises at `u ≠ 0`
  and then sees the amplitude term), `fd_scheme` central/forward, `fd_step_m`, `jac_mode`, `huber_delta_frac`,
  `radius_init_m`, `rho_low/rho_high/shrink/grow`, `at_radius_frac`, `use_warm_start`.
- `adam`: `lr_m`, `lr_decay`, `init_noise_m`, `grad_mode`, `max_iters`.
- Both: `horizon`, `discount`, `budget_safety`, `time_factor`.

## What the toy showed (`toy.py`, P = 2000, goal 0.28 m, bound 0.30 m, 5 ticks, H = 1, budget 0.3 s, CPU, 3 toy seeds)
| progress `1 − D_5/D_0` | adam | trgn | trgn, 1 inner step | random shooting |
|---|---|---|---|---|
| model = world | 1.000 | 1.000 | 1.000 | 0.981 |
| defect G = 0.25, β = 1.25 /m | 0.427 | 0.924 | 0.752 | 0.450 |

Defect: directional response × G plus a pull toward the goal ∝ commanded amplitude, so toward-goal predicted change is
0.6 of realized and away/lateral moves are predicted to lower the cost. Trusting that model cost (adam, random
shooting) overshoots every tick near the goal (step norms 0.30, 0.15, 0.28, 0.30, 0.30; ρ 1.3, −1.3, −0.6, 0.2, −0.3).
`trgn` radius trace, seed 0: defect off 0.05, 0.10, 0.20, 0.20, 0.10; defect on 0.05, 0.10, 0.20, 0.30, 0.15 with ρ 1.68,
1.87, 0.78, −1.03, 1.83. The radius caps the step but does not rescale it: at 7 cm from the goal with Δ = 0.30 the GN
step (∝ 1/G) overshot to 13 cm before ρ < 0 halved Δ. A gain estimate from ρ applied to the step length is the obvious
next variant; not implemented. Per plan on the toy: trgn 15–40 ms and 1–57 rollouts; adam about 375–500 iterations.

## Unverified against the real predictor
- Everything about behaviour with the 22M model: progress, ρ statistics, whether the 5 cm initial radius and the ×2/×0.5
  rule suit 5 ticks, whether the toy defect resembles the real one beyond the E001 signs it was built from.
- `fd_step_m = 1 mm`: E001 found the model's odd response had the wrong sign in 7/8 instances at 3 mm and the right sign
  at 1–3 cm. The FD step may need to be 1–3 cm on the real model; FP32 noise at 1 mm over 1e6 features is also untested.
- `jvp` through the predictor's attention kernels was never run; only the toy (tanh, matmul). Forward mode showed a
  one-off first-use cost of about 1 s per process on the toy, which the planner cannot keep inside a budget — the harness
  should issue one untimed warm-up `plan()` if `jac_mode='jvp'` is used. Reverse mode through the real rollout: untested.
- Timing: the budget guard uses the durations of the previous linearisation and evaluation (×1.5, 10% reserve). The
  first rollout of each `plan()` is unconditional, so a budget shorter than one FD batch (6H+1 sequences) is exceeded.
  GPU timing, memory of `J` (H·P·3H floats; 12 MB at H = 1, P = 1e6, quadratic in H) and of the 6H+1 batch: unmeasured.
- Budget-limited loops are deterministic only up to the number of iterations that fit; `max_iters` /
  `max_inner_iters` make them exact (tested that way).
- `ρ` is measured on the native L1 cost while `trgn` minimises the pseudo-Huber surrogate; `ρ = None` (zero predicted
  change, e.g. a rejected plan of zero motion) shrinks Δ, per the spec.
