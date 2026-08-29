# Certified resolved-response prefactor and final Phase 2 aggregation

Resolved-response producer: blaschke_deformation_phase2_resolved_response.py; final aggregation: blaschke_deformation_phase2_final_aggregation.py; map: blaschke_mu_0p3.
N=600, M=610, rho=2.725, r=2.473669807791324.
Arb precision: 192 bits; boundary cells: 65536; coherent prefix: 24 modes.

Coherent Chebyshev-packet response upper: 2.12513094458366014e+00.
Scaled-Legendre response upper: 3.57658064614203795e+02.
Whole-ellipse finite-restriction fallback: 1.29484356379656571e+01.
Selected cellwise response upper: 2.12513094458366014e+00.
Certified output-tail contribution B_out: 2.8540338522337406e-23.
Final selected input contribution B_in: 3.3264421486719304e-20.
Input selection: coherent_branchwise_intersection.
The selected unresolved-input intersection equals the branchwise upper on every deployed boundary cell; no cancellation gain is claimed.
Final selected matrix contribution: 1.0872470878179423e-28.
Matrix selection: starred.
Final deterministic epsilon upper: 0.000000000000000000033264433839017426342179265983234893950031511766904056153485116.
Auxiliary triangle epsilon upper: 0.000000000000000000033292961933966350187794230000000000000000000000000000000019506.

The provisional resolved-response radius has been replaced by the authoritative standalone aggregation.

Upstream SHA-256 hashes:
- `branch_image_balanced_candidate_single_space_row_N600_M610_safe_GL.csv`: `abd6b2f884729c53875c26c3b9b0c09f449d1fc4fd72a33e15465ba91feac91f`
- `branch_image_balanced_candidate_transport_cert_N600.csv`: `6f4a6c124b0247c3957fc03fb4ae7f338abcf637ef811e3212753c91f78d96ba`
- `branch_image_radius_reoptimisation_balanced_highcell_scan.csv`: `85ed4ba755d808884391de5f7208ae1a126bcad6a8a92602001dae2cfe57afbe`
- `blaschke_deformation_certification.py`: `dbe80820b21637e2cd3f24e28f6c4ee0288af81197af40a4e6871511bfa17501`
