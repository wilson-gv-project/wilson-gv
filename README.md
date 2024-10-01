# Wilson

1. `main2DIR.py` contains implementations (rendering and expression implementations) of the infrastructure for 2D-IR spectrum simulation which are then used in `testmain.py`.
2. `pyrsp_2dir.py` contains calculations of properties with PyOpenRSP (some functions of `main2DIR.py` still depend on the output data in the format of LSDalton calculation).
3. `openrsp_tensor_reader.py` is now mixed with PyOpenRSP data format post-processing.
4. `testmain.py` contains the simulation protocol with all the steps.
