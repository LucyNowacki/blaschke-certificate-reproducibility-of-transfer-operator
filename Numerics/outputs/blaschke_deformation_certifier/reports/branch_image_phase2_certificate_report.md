# Phase 2 branch-image certificate report

Promoted row: N=600, M=610.
Branch-image radii: rho=2.725, r=2.473669807791324, r_tau=2.2930919118225574493.
q_out approximately 0.90776873680415563645; certified q_gap <= 0.927; certified q_star <= 0.927.

## Bound components

- output tail B_out: 2.8540338522337406e-23
- coherent packet response upper: 2.12513094458366
- scaled-Legendre response upper: 357.6580646142038
- whole-ellipse response fallback: 12.948435637965657
- selected response upper: 2.12513094458366
- response selection: cellwise minimum of coherent packet, scaled Legendre, and whole-ellipse fallback certificates
- response prefix and cells: 24 modes over 65536 cells
- selected input tail: 3.3264421486719304e-20
- input selection: coherent_branchwise_intersection
- selected coherent-branchwise intersection upper: 3.3264421486719304e-20
- branchwise input upper: 3.3264421486719304e-20
- best transported collocation matrix defect: 1.0872470878179423e-28
- matrix selection: starred
- total epsilon_X upper: 0.000000000000000000033264433839017426342179265983234893950031511766904056153485116
- auxiliary triangle epsilon_X upper: 0.000000000000000000033292961933966350187794230000000000000000000000000000000019506
- response prefactor certified: True
- finite-M quadrature prefactor certified: True
- complete row theorem-certified: True

For the unresolved input, the selected coherent-branchwise intersection equals the branchwise upper on every boundary cell; the deployed row therefore demonstrates no unresolved-tail cancellation gain.  On the resolved-output side, the cellwise coherent packet certificate is sharper than the scaled-Legendre and whole-ellipse fallbacks before the common output-tail factor is applied.