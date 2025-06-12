# wilson-suite

1. Get required repos with `setup.sh`:
   - wilson_main
   - wilson_utils
   - wilson_derive
   - wilson_experiment
   - wilson_intensities
   - wilson_analysis
   - CQCParse
2. `pip install -r requirements.txt` - install obtained packages.
3. Add to Python path python module `wilson_suite.py`. Aliases are defined there, so `import wilson_suite.main` is possible.

First 2 steps automate/facilitate downloading and installation of repos (now there is one option of getting all of them but can be more flexible). The last step isn't an installation but it makes possible nested imports.

The content of `wilson_suite.py`:
```python
import wilson_analysis as wilson_analysis
import wilson_main as wilson_main
import wilson_derive as wilson_derive
import wilson_experiment as wilson_experiment
import wilson_utils as wilson_utils
import wilson as wilson_intensities
```
