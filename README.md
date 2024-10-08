# Wilson

## Next steps - features
- [ ] Coriolis for GVPT2
- [ ] 3 quanta levels
- [ ] GVPT2 for freqs
- [ ] dynamic range parameters and colorbar scale - figure out rendering
- [ ] optimize intensity calculation - minimize redundant calculations


## Big picture

| Section           | Description                                      | Extra (for paper II) | Module |
|-------------------|--------------------------------------------------|----------------------|--------|
| Parsing           | CFOUR and Gaussian outputs parsing               | Class template       |        |
| Intensities       | Spectrum class, Expressions class?? calculations |                      |        |
| Orient. averaging | Derivatives data, averaged tensor                |                      |        |
| Rendering         | 2D spectrum plot                                 |                      |        |

---
### Data analysis tools

| Section                               | Description                                                |
|---------------------------------------|------------------------------------------------------------|
| Elec. vs Mechanical anharmonicities   | Analysis of these contributions, visual/table              | 
| Normal modes view (by label, freq.)   | Visualization of vibrations, normal mode vector            | 
| Contributions of modes to signals     | Table? maybe combined with "simplified spectrum"           |
| Overview of resonances                | "Simplified spectrum", table                               |
| Printing derivs data, averaged tensor | Tables? to show which modes have significant contributions |
| Outcomes of the paper I               | Accessing characteristic features of spectrum              | 
