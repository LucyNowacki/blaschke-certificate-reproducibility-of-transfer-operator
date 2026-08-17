# Phase 3 branch-image summary

## Branch-image deterministic row

- N = 600, M = 610
- rho = 2.725
- r = 2.473669807791324
- r_tau = 2.293091911822557
- q_star = 0.927
- epsilon_X = 3.326443383901743e-20
- epsilon_X divided by q_star^N = 1.87992121720786532100821557548

## Empirical raw spectral behaviour

The raw curves use only the square Legendre--Gauss transfer blocks Lhat_N^(N).  They are diagnostic spectral errors, not theorem-level operator-norm bounds.

- mu^1: fitted base 0.03425, fitted base divided by q_star 0.0369471, N range 10 to 100
- mu^2: fitted base 0.0799406, fitted base divided by q_star 0.0862358, N range 10 to 100
- alpha^1: fitted base 0.218252, fitted base divided by q_star 0.235439, N range 10 to 100
- alpha^2: fitted base 0.227461, fitted base divided by q_star 0.245373, N range 10 to 100
- alpha^3: fitted base 0.232859, fitted base divided by q_star 0.251196, N range 10 to 100
- alpha^5: fitted base 0.277177, fitted base divided by q_star 0.299004, N range 10 to 30

## Deterministic interpretation

The response-prefactor branch-image row is far smaller than the old whole-ellipse row.  At N=600, M=610 the deterministic radius is controlled by the analytic single-space tails, while the transported Schur matrix defect is negligible.

The branch-image q_star is the comparison base used by the later contour stage.  Phase 4 should use the exported compatibility row from Phase 2, namely phase2_certified_single_space_row_N600_M610.csv.