# Phase 2 branch-image certificate report

Promoted row: N=600, M=610.
Branch-image radii: rho=2.725, r=2.473669807791324, r_tau=2.293091911822557.
q_out=0.9077687368041556, q_gap=0.92700000000000005, q_star=0.92700000000000005.

## Bound components

- output tail B_out: 2.8540338522337406e-23
- coherent packet response upper: 2.12513094458366
- scaled-Legendre response upper: 357.65806461420146
- whole-ellipse response fallback: 12.948435637965648
- selected response upper: 2.12513094458366
- response selection: cellwise minimum of coherent packet, scaled Legendre, and whole-ellipse fallback certificates
- response prefix and cells: 24 modes over 65536 cells
- selected input tail: 3.3264421486719304e-20
- input selection: coherent_row
- coherent-row input upper: 3.3264421486719304e-20
- branchwise input upper: 3.3264421486719304e-20
- best transported Schur matrix defect: 1.0872470878179354e-28
- matrix selection: starred
- total epsilon_X: 3.32644338390174263421785759832e-20
- auxiliary triangle epsilon_X: 3.329296193396635018779354e-20
- response prefactor certified: True
- finite-M quadrature prefactor certified: True
- complete row theorem-certified: True

The branch-image row selects the best supplied coherent, branchwise and whole-ellipse input certificate.  On the output side it selects cellwise among the coherent packet, scaled-Legendre and whole-ellipse resolved-response certificates before applying the common output-tail factor.