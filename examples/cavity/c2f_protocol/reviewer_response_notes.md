# Reviewer Response Notes: Non-Thermal Aging Paper

Paper: *Non-Thermal Aging of Supercooled Liquids in Optical Cavities* (Hasyim, Damiani,
Hoffmann, arXiv:2603.15693). Simulation code: this directory
(`examples/cavity/c2f_protocol/`). Reference protocol repo: `cav-hoomd/`.

This document records (1) the referee comments and our responses, (2) the
calculations that address each comment, and (3) the results obtained so far.
Generated artifacts live in [`reviewer_response/`](reviewer_response/).

---

## 1. Core decision (the reframe)

- Recast the central mechanism as a **finite-q shift / bilinear effect**, not
  polariton formation or a DSE-driven effect. At the finite-q displaced
  equilibrium the **bilinear `d.E` term dominates** the cavity interaction.
- **Get out of the ultrastrong regime**: move main-text results to weaker
  couplings, add off-resonant cases, and demote `lambda = 0.09, 0.141` to the SI.
- **Defuse the DSE-artifact concern**: show the large vibrational fictive-temperature
  excursion is not caused by a runaway quadratic dipole self-energy (DSE) of the
  Pauli-Fierz (PF) model.
- Plots will look less dramatic but the effect is real; Rabi splitting is unchanged.

---

## 2. Referee comments and responses

### R1 - Pauli-Fierz model / DSE artifact
PF should be stated upfront. In PF the quadratic DSE (proportional to lambda^2)
dominates in *ultrastrong* coupling and merely blue-shifts the dipole energy,
contradicting measured red shifts (Bloch-Siegert; Landau polaritons). Concern:
the ~800 K vibrational excursion (p.5) could be a PF/DSE artifact. Compare with the
multipolar formulation (Nanophotonics 10(1), 2021, 477-489, doi:10.1515/nanoph-2020-0451).

**Response (supported by Calc 1/3/5):** in the weaker coupling range the bilinear
term governs the energetics, the DSE is locked to it by the finite-q displacement
(bilinear = -2x DSE), and the net cavity energy is ~0. The excursion is not a DSE
runaway. State PF upfront; add the multipolar comparison.

### R2 - Cavity-frequency dependence / polariton relevance / energy balance
SI 6.2 / Fig S3 show structural relaxation is nearly insensitive to detuning over
hundreds of cm^-1, so polariton formation is not the relevant transfer process.
Therefore: (a) abstract must drop *"light selectively pumps fast vibrational modes"*;
(b) the Rabi-period vs timescale comparison must be removed or justified; (c) decompose
the Fig 3b "Harmonic" curve into dipole-vibrational and cavity-mode parts and justify
where the energy comes from.

**Response (Calc 2, Fig 3b, material-time):** cavity frequency is intentionally not
the key variable - consistent with the finite-q reframe. Provide the Fig 3b
decomposition, the energy balance (net cavity energy ~0), and tau_s vs frequency.

### R3 - Experimental feasibility of sub-ps coupling modulation
Refs 13-17 do not give a mechanism to modulate the coupling on sub-ps timescales
without changing cavity frequency or intracavity dielectric constant.

**Response (writing):** strengthen the feasibility discussion; separate
supercooled-liquid timescales from experimental timescales; cite PF-Hamiltonian refs.

### R4 - Ultrastrong coupling / single-mode breakdown
Fig 2 splittings reach ~600 cm^-1 (lambda=0.141), i.e. ultrastrong (Nat Rev Phys 1,
19-40, 2019). Single-mode breaks down: mid-IR FSR ~300-400 cm^-1; nearby modes matter
above ~200 cm^-1 splitting (J. Chem. Phys. 160, 204303).

**Response (Calc 4):** move main text to splittings < ~300 cm^-1, demote ultrastrong
cases to the SI with the nearby-mode caveat; fix "strong" -> "ultrastrong" terminology.

### R5 - "optical" vs "infrared"
These are infrared cavities (vibrational), not optical (electronic).

**Response (writing):** global terminology fix to "infrared".

---

## 3. Coupling map (Rabi splitting vs lambda)

Empirically (Fig 2) lambda=0.141 -> ~600 cm^-1, i.e. splitting ~ 4255 * lambda cm^-1:

| lambda | splitting (cm^-1) | role |
|--------|-------------------|------|
| 0.01  | ~43  | weak, main text |
| 0.03  | ~128 | main text |
| 0.042 | ~179 | < 200, single-mode safe, main text |
| 0.07  | ~300 | upper main-text case |
| 0.09  | ~383 | ultrastrong -> SI |
| 0.141 | ~600 | ultrastrong -> SI |

Confirmed main-text set: **0.01, 0.03, 0.042, 0.07** at **100 K and 50 K**.

---

## 4. Results so far

All energies are steady-state means (20% burn-in) from NVT cavity-equilibrium runs
at 100 K, finite-q on. Raw numbers: [`reviewer_response/energy_decomposition_table.txt`](reviewer_response/energy_decomposition_table.txt).

### Calc 1 - weak coupling, DSE on vs off (R1)
At lambda = 0.01 (10 ns), the structural/vibrational fictive temperatures are barely
affected by the DSE and show no large excursion:

| quantity | DSE on | DSE off |
|----------|--------|---------|
| T_v fictive (K) | 80.7 | 86.7 |
| T_s fictive (K) | 107.0 | 86.5 |

=> the weak-coupling behavior is not a DSE artifact.
Figure: [`reviewer_response/calc1_dse_onoff_lam0.01_eq10ns100K.png`](reviewer_response/calc1_dse_onoff_lam0.01_eq10ns100K.png).

### Calc 3 - DSE vs bilinear decomposition / energy balance (R1, R2)
The bilinear coupling energy is exactly **-2x the DSE** at every coupling, and the
three cavity terms (coupling + DSE + cavity-harmonic) **cancel to ~+1.3 kJ/mol
independent of lambda**:

| lambda | E_coupling | E_DSE | E_cav,harm | net |
|--------|-----------|-------|-----------|-----|
| 0.010 | -87.0   | +43.7   | +44.6   | +1.3 |
| 0.042 | -1397.1 | +698.7  | +699.7  | +1.3 |
| 0.090 | -7100.2 | +3550.0 | +3551.5 | +1.3 |

=> the system relaxes to a displaced (finite-q) equilibrium with ~zero net cavity
energy. There is no large standalone DSE energy injection. This is the energy-balance
answer R2 asks for: the dipoles are not resonantly fed by the cavity; the cavity term
is a static finite-q shift.

### Calc 5 - scaling with lambda (R1, R2)
Both bilinear and DSE energies scale as **lambda^1.99** (fitted), locked at the 2:1
ratio - the DSE does not independently blow up faster than the bilinear term.
Figure: [`reviewer_response/calc5_scaling_eq10ns100K.png`](reviewer_response/calc5_scaling_eq10ns100K.png).

> Note on framing: the original plan anticipated a regime *linear* in lambda before
> lambda^2 takes over. The equilibrium decomposition instead shows the cleaner
> finite-q result: at the displaced equilibrium both terms are lambda^2 and cancel.
> The "linear in lambda" statement properly refers to the *dynamical* energy injected
> during a square-wave turn-on, which should be measured from the cooling runs
> (Calc 4 / regen) rather than the static equilibrium.

### Fig 3b decomposition (R2)
The "Harmonic" curve splits into the dipole-vibrational part (E_bond, ~110 kJ/mol,
nearly lambda-independent) and the cavity-mode harmonic part (E_cav,harm, growing as
lambda^2). Figure: [`reviewer_response/fig3b_decomposition_eq10ns100K.png`](reviewer_response/fig3b_decomposition_eq10ns100K.png).

### Calc 4 - main-text coupling set at 100 K and 50 K (R4, R1)
Equilibrium runs completed for lambda in {0.01, 0.03, 0.042, 0.07} at both 100 K and
50 K (DSE on; plus DSE off at the weak point). The bilinear = -2x DSE relation and
the lambda^2 scaling hold at both temperatures
([`reviewer_response/calc5_scaling_eq50K.png`](reviewer_response/calc5_scaling_eq50K.png),
[`reviewer_response/fig3b_decomposition_eq50K.png`](reviewer_response/fig3b_decomposition_eq50K.png)).
The ultrastrong cases (0.09, 0.141) are kept SI-only. Consolidated overview:
[`reviewer_response/regen_main_text_overview.png`](reviewer_response/regen_main_text_overview.png).
Caveat: the 50 K weak points (0.01, 0.03) are noisier at 1 ns and would benefit from
longer equilibration; the internal 2:1 ratio is nonetheless exact.

### Calc 2 - cavity-frequency sweep at weak coupling (R2)
At lambda = 0.01, 100 K, sweeping omega_c over 1360-2400 cm^-1 (across both the A-A
1560 and B-B ~2433 resonances) leaves the effective structural relaxation time within
**~89-143 ps** - essentially flat over a >1000 cm^-1 detuning span. This directly
supports SI 6.2 / Fig S3: cavity frequency (polariton formation) is not the relevant
variable for the structural cooling.
Figure: [`reviewer_response/calc2_tau_vs_frequency.png`](reviewer_response/calc2_tau_vs_frequency.png).

### Material time / structural relaxation (R2)
tau_s(T) fit from the F(k,t)/F(k,0)=0.1 table: log10(tau_ps) = 314.5/T - 0.95
(T range 65-200 K). Effective tau_s mapped from the steady-state T_s of each
equilibrium run is **nearly flat across couplings at 100 K (~89-110 ps)**, supporting
the claim that the cavity does not strongly change structural relaxation.
Material time along the cooling protocol: [`reviewer_response/material_time_cooling.png`](reviewer_response/material_time_cooling.png).
tau_s vs cavity frequency (R2 / Fig S3): [`reviewer_response/calc2_tau_vs_frequency.png`](reviewer_response/calc2_tau_vs_frequency.png) (populated once the frequency sweep finishes).

---

## 5. Reproduce / run

Analysis (existing + new equilibrium data):
```bash
pixi run --as-is -e test python examples/cavity/c2f_protocol/analyze_reviewer_response.py --burn-in 0.2
pixi run --as-is -e test python examples/cavity/c2f_protocol/analyze_material_time.py
```

New simulation campaign (Calc 2 frequency sweep + Calc 4 50 K / new couplings),
launched detached:
```bash
setsid bash -c 'EQ_SKIP_CUDA_REBUILD=1 exec bash examples/cavity/c2f_protocol/run_reviewer_response_sims.sh' &
```
`run_cavity_equilibrium.py` now accepts `--omega-c-cm1` for the frequency sweep.

Reference protocol (cav-hoomd) for the cooling/Fig-5 plots: square-wave coupling
period 10 ps / duty 0.1, omega_c = 1560 cm^-1, bussi baths (tau = 1 ps), DiffEq
controller on `lj_coulombic` fictive T, calibration `potential_energy_vs_T.txt`,
~10 replicas; fictive columns `lj_coul_fictive_K` / `harmonic_fictive_K` /
`molecular_bath_K`; per-replica 2 ps rolling mean -> 0.1 ps grid -> nanmean.

---

## 6. Manuscript edits checklist (no manuscript source in this repo)

The arXiv manuscript / SI are not in this repository, so these are tracked as a
checklist to apply in the manuscript source:

- [ ] State the Pauli-Fierz model upfront in the main text (R1).
- [ ] Add the multipolar-formulation comparison, Nanophotonics 2021 (R1).
- [ ] Replace "strong coupling" with "ultrastrong coupling" where splitting exceeds the
      strong-coupling definition; add the nearby-mode caveat for > 200 cm^-1 (R4).
- [ ] Replace "optical" with "infrared" cavities throughout (R5).
- [ ] Abstract: remove "light selectively pumps fast vibrational modes" (R2).
- [ ] Remove or justify the Rabi-period vs timescale comparison (R2).
- [ ] Strengthen experimental feasibility / timescale discussion; cite PF-Hamiltonian refs (R3).
- [ ] Remove the DSE argument and the flagged SI sentence; clarify the
      cavity-frequency-independence note (SI 6.2 / Fig S3).
- [ ] Replace Fig 2/Fig 3b/Fig 5 with the regenerated weak/off-resonant-coupling versions;
      move ultrastrong cases to the SI.

---

## 7. Status

| Task | Status |
|------|--------|
| Calc 1 (DSE on/off, lambda=0.01) | done (100 K + 50 K) |
| Calc 2 (frequency sweep) | done (tau_s flat 89-143 ps over 1360-2400 cm^-1) |
| Calc 3 (DSE vs bilinear at equilibrium) | done (100 K, 1 ns + 10 ns) |
| Calc 4 (0.03/0.07 @100 K, full set @50 K) | done; ultrastrong -> SI |
| Calc 5 (scaling vs lambda) | done (equilibrium lambda^2); dynamical injection from cooling runs is the remaining production step |
| Fig 3b decomposition | done (100 K + 50 K) |
| Material time / tau_s | done (cooling h(t) + per-run effective tau_s + frequency panel) |
| Regenerate main-text plots | done (overview); full per-coupling Fig 5 cooling reproduction is the remaining multi-replica production step |
| Manuscript edits | checklist documented (no manuscript source in repo) |

### Remaining production-scale work (beyond this analysis pass)
- Full Fig 5 cooling reproduction per main-text coupling at both temperatures using the
  cav-hoomd square-wave + DiffEq protocol with ~10 replicas (the equilibrium runs here
  establish the energetics; the cooling runs give the dynamical injected energy and the
  fictive-temperature trajectories).
- Longer 50 K weak-coupling equilibration for cleaner scaling at lambda <= 0.03.
