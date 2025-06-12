# wilson-suite

After cloning this repo, `cd wilson-suite`:
1. Get required repos with `setup.sh`:
   - wilson_main
   - wilson_utils
   - wilson_derive
   - wilson_experiment
   - wilson_intensities
   - wilson_analysis
   - CQCParse
2. `conda env create -f environment.yml` - set up an environment; packages will be installed.
3. `export PYTHONPATH="${PYTHONPATH}:$(pwd)"` - add to Python path directory with Python module `wilson_suite.py`. Aliases are defined there, so `import wilson_suite.main` is possible.
4. `python -c "import wilson_suite"` - test if it works.


First 2 steps automate/facilitate downloading and installation of repos (now there is one option of getting all of them but can be more flexible). The third step isn't an installation but it makes possible nested imports.

The content of `wilson_suite.py`:
```python
import wilson_analysis as analysis
import wilson_main as main
import wilson_derive as derive
import wilson_experiment as experiment
import wilson_utils as utils
import wilson as intensities
```
