#### Data for the 2D IR spectrum

### Frequencies
1. **Vibrational analysis of the equilibrium hessian.**
   a. Mass-weighting hessian
   b. Projecting out rotational and translational modes of hessian
2. **VPT2 treatment**: getting anharmonic corrections (xguinea)
   a. Cubic and quartic forces (displaced hessians)

### Dipole moment derivatives (1st and 2nd)
Tasks:
1. Compare DIPDER and Dipole Moment Function in outfile **--- +++** aren't the same... Conclusion: hmm
2. Run PROPS=FIRST_ORDER  **--- +++** Conclusion: DIPDER data is in VIB=ANALYTIC calculation
3. Numerically compute the 1st derivative of dipole moment and compare it to DIPDER or Dipole Moment Function 


### Polarizability derivatives (1st and 2nd)
(same as for cubic and quartic forces?)

### Cubic forces 
(Displaced hessians)

## Scripts

I. Vib. analysis for equil. hessian ()


### Notes now:
- `methanol/numdiffpol1_h4` - `delta` was `0.001`
- `methanol/numdiffhess4` - `delta` was probably also `0.001`

------

### Fast track:

#### I. CALC=CCSD(T), BASIS=cc-pVDZ

*CFOUR(CALC=CCSD(T), BASIS=cc-pVDZ
ABCDTYPE=AOBASIS
CC_PROG=ECC
GEO_CONV=10
CC_CONV=10
SCF_CONV=10
LINEQ_CONV=10
VIBRATION=ANALYTIC
ANHARM=VPT2
DROPMO=1
ANH_ALGORITHM=PARALLEL
FD_PROJECT=OFF
PRINT=1
COORD=CARTESIAN,UNITS=BOHR
MEMORY_SIZE=4
MEM_UNIT=GB)

#### II. CALC=HF, BASIS=cc-pVDZ

*CFOUR(CALC=CCSD(T), BASIS=cc-pVDZ
ABCDTYPE=AOBASIS
CC_PROG=ECC
GEO_CONV=10
CC_CONV=10
SCF_CONV=10
LINEQ_CONV=10
VIBRATION=ANALYTIC
ANHARM=VPT2
DROPMO=1
ANH_ALGORITHM=PARALLEL
FD_PROJECT=OFF
PRINT=1
COORD=CARTESIAN,UNITS=BOHR
MEMORY_SIZE=4
MEM_UNIT=GB)

