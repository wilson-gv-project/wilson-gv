## 2D IR intensities - Units

dq Normal coordinates - dimensionless

### Dipole moment - `C·m // Debye // a.u.`
One debye corresponds to 10^−21 C·m2/s divided by the speed of light. 
Conversely, 1 C·m ≘ 2.9979×10^29 D
1 au = e * a_0
(Debye = 10**-18 statcoulomb cm , SI units = C m)

### Dipole moment derivatives (1) - `e·a_0/(amu·a_0^2) // a.u./(amu·a_0^2) // e`


### Dipole moment derivatives (2) - `?? // e/a_0`


### Polarizability - `C·m^2·V^−1 // a.u. // e^2·a_0^2`


### Polarizability derivatives (1) - ` // e^2·a_0/E_h`


### Polarizability derivatives (2) - ` // e^2/E_h`


### Cubic force constants - `E_h/a_0^3 // cm^-1`


|  Property	  |     SI 	      |     a.u. 	      | dq	 (dimensionless nc) |                        dQ (mass-weighted nc)	                         |  	esu   |
|:-----------:|:-------------:|:---------------:|:----------------------:|:---------------------------------------------------------------------:|:-------:|
|    	`mu`    |     `C·m`     | `a.u. = e·a_0`  |     `a.u. = e·a_0`     |                       !  `Debye`  !          	                        |    	    |
|  	  `d_mu`  |    `C`  ?     |       `e`       |          	`e`          |    `a.u./(amu*a_0^2) = e/(amu*a_0)` !  `Debye/(amu^(1/2)*a0)`  ! 	    |    	    |
| 	 `d^2_mu`  |    `C/m` ?    |        	        |           	            |             ! `Debye/(amu*a0^2)` !                      	             |    	    |
|      	      |       	       |        	        |           	            |                                   	                                   |    	    |
|   `alpha`   | `C·m^2·V^−1`  | `e^2·a_0^2/E_h` |    `e^2·a_0^2/E_h`     |                        !`a0^3`!             	                         | 	`cm^3` |
| 	`d_alpha`  | `C·m·V^−1` ?  |  `e^2·a_0/E_h`  |     `e^2·a_0/E_h`      |                  !`a0^2/amu^(1/2)`!                	                  |    	    |
| `d^2_alpha` |  `C·V^−1` ?   |    `e^2/E_h`    |       `e^2/E_h`        |                       !`a0/amu`!              	                       |    	    |
|      	      |       	       |        	        |           	            |                                   	                                   |    	    |
| 	  `F_ijk`  |    `J/m^3`    |  	 `E_h/a_0^3`  |      `E_h/a_0^3`       |                   `E_h/(amu^(3/2)·a_0^3)`        	                    | `cm^-1` |
|      	      |       	       |        	        |           	            |                                   	                                   |    	    |
|   	  `dQ`   | `sqrt(kg)·m`	 | `sqrt(amu)·a_0` |           	            |                        ! `amu^(1/2) a0`    ! 	                        |    	    |
|   	  `dq`   |     ` `	      |        	        |           	            |                                   	                                   |    	    |


```
 :        CUBIC FORCE CONSTANTS IN NORMAL MODES         :
 :                                                      :
 : FI =  Reduced values [cm-1]  (default input)         :
 : k  =  Cubic Force Const.[AttoJ*amu(-3/2)*Ang(-3)]    :
 : K  =  Cubic Force Const.[Hartree*amu(-3/2)*Bohr(-3)] :
```


| Base Quantity             | Symbol | Dimensional Formula |
|---------------------------|--------|---------------------|
| Length                    | L      | `[L]`               |
| Mass                      | M      | `[M]`               |
| Time                      | T      | `[T]`               |
| Electric Current          | I      | `[I]`               |
| Thermodynamic Temperature | Θ      | `[Θ]`               |
| Amount of Substance       | N      | `[N]`               |
| Luminous Intensity        | J      | `[J]`               |


| Physical Quantity                                         | Formula          | Dimensional Formula              | S.I Unit                             |
|-----------------------------------------------------------|------------------|----------------------------------|--------------------------------------|
| Energy (Quantum Systems)                                  | `E = hν`         | `[M][L]^2[T]^{-2}`               | `Joule (J)`                          |
| Planck's Constant                                         | `h`              | `[M][L]^2[T]^{-1}`               | `Joule second (Js)`                  |
| Angular Momentum                                          | `L = Iω`         | `[M][L]^2[T]^{-1}`               | `Joule second (Js)`                  |
| Momentum                                                  | `p = mv`         | `[M][L][T]^{-1}`                 | `Kilogram meter per second (kg m/s)` |
| Wavelength                                                | `λ = h/p`        | `[L]`                            | `Meter (m)`                          |
| Frequency                                                 | `ν`              | `[T]^{-1}`                       | `Hertz (Hz)`                         |
| Wave Number                                               | `k = 2π/λ`       | `[L]^{-1}`                       | `Per meter (m^{-1})`                 |
| Electric Field                                            | `E`              | `[M][L][T]^{-3}[I]^{-1}`         | `Volt per meter (V/m)`               |
| Magnetic Field                                            | `B`              | `[M][T]^{-2}[I]^{-1}`            | `Tesla (T)`                          |
| Electron Spin                                             | `S`              | `Dimensionless (Quantum Number)` | `Dimensionless`                      |
| Quantum Wavefunction                                      | `ψ`              | `Dimensionless`                  | `Dimensionless`                      |
| Electric Dipole Moment                                    | `p = qd`         | `[I][T][L]`                      | `Coulomb meter (C·m)`                |
| Probability Density                                       | `ρ = \|ψ   \|^2` | `[L]^{-3}`                       | `Per cubic meter (m^{-3})`           |
| Polarizability                                            | `α`              | `[L]^3`                          | `Cubic meter (m^3)`                  |
| First Derivative of Dipole Moment wrt Normal Coordinate   | `∂μ/∂Q`          | `[I][T][L][M]^{-1/2}[L]^{-1/2}`  | `Coulomb meter (C·m)`                |
| Second Derivative of Polarizability wrt Normal Coordinate | `∂²α/∂Q²`        | `[L]^3[M]^{-1}[L]^{-1}`          | `Cubic meter per meter (m^2)`        |

| Electromagnetic Quantity | Dimensional Formula in SI  | SI Unit                    | Dimensional Formula in a.u. | a.u. Unit                                |
|--------------------------|----------------------------|----------------------------|-----------------------------|------------------------------------------|
| Electric Charge          | `[I][T]`                   | Coulomb (C)                | `Dimensionless`             | Elementary charge (e)                    |
| Electric Dipole Moment   | `[I][T][L]`                | Coulomb meter (C·m)        | `[L]`                       | Bohr radius (a₀ e)                       |
| Electric Field           | `[M][L][T]^{-3}[I]^{-1}`   | Volt per meter (V/m)       | `[L]^{-1}[E_h][e]^{-1}`     | Hartree per Bohr radius (E_h/(a₀ e))     |
| Electric Potential       | `[M][L]^2[T]^{-3}[I]^{-1}` | Volt (V)                   | `[E_h][e]^{-1}`             | Hartree per electron charge (E_h/e)      |
| Magnetic Field           | `[M][T]^{-2}[I]^{-1}`      | Tesla (T)                  | `[L]^{-1}[T]^{-1}`          | Bohr magneton per Bohr squared (µ_B/a₀²) |
| Magnetic Dipole Moment   | `[I][L]^2`                 | Ampere square meter (A·m²) | `[L]^2[T]^{-1}`             | Bohr magneton (µ_B)                      |
| Polarizability           | `[L]^3`                    | Cubic meter (m³)           | `[L]^3`                     | Cubic Bohr radii (a₀³)                   |