# GeochemistryPi — Chemical Modeling Module

This module (chemical modeling) provides two kinds of routines:
- Internal standard method (currently: Hg)
- Double-spike method (currently: Mo)

It is an interactive command-line tool located under `src/` and designed to read inputs from `data/` and write outputs to `results/`.

---

## For users

Prerequisites (macOS)
- Python 3.8+
- Recommended virtual environment:
  - python3 -m venv .venv
  - source .venv/bin/activate
- Install dependencies:
  - pip install -r requirements.txt
  If not present, at minimum install: pandas, numpy, scipy, openpyxl, xlsxwriter

Quick start
1. Put input Excel files under `data/`:
   - `data/Hg_data.xlsx` for Hg internal-standard method
   - `data/Mo_data.xlsx` (sheet `3程序处理_输入常数`) for Mo double-spike method
2. Run the launcher:
   - From project root: `python src/main.py`
3. Follow prompts:
   - First choose method:
     - `1` Internal standard method
     - `2` Double spike method
   - Then choose element for the selected method:
     - If Internal standard → `1` (Hg)
     - If Double spike → `1` (Mo)
4. Results are written to `results/`:
   - Hg internal-standard: `results/Hg_results.xlsx` (or script-specific outputs)
   - Mo double-spike: `results/Mo_results.csv`

Notes
- The launcher executes element-specific scripts or functions. Errors during execution are printed to the console.
- Keep input sheet/column names as expected by the scripts (see data examples).

---

## For developers

Goal
- Make it easy to add new elements and methods while keeping the launcher UI intact:
  - First select method → then select element(s) available under that method.

Project structure (relevant)
- src/
  - main.py                     — interactive launcher and core solvers
  - Hg_Internal_standard_method.py — Hg internal-standard implementation
  - (future) <Element>_Internal_standard_method.py
  - (future) <Element>_Double_spike.py
- data/                          — input Excel files
- results/                       — outputs
- tests/                         — unit tests (recommended)

How to add a new element (internal standard method)
1. Create a new script:
   - `src/<Element>_Internal_standard_method.py`
   - The script should be runnable standalone (executable by runpy.run_path) and locate its input via:
     ```python
     excel_path = os.path.join(os.path.dirname(__file__), '..', 'data', '<Element>_data.xlsx')
     ```
2. Implement:
   - Basic input validation (required columns, types).
   - Core processing function(s).
   - Save outputs into `results/` with a clear filename (e.g., `<Element>_results.xlsx`).
3. Wire it into the launcher:
   - In `src/main.py`, add the element to the Internal Standard branch:
     - Add a menu entry (simple prompt text).
     - Call the script via the helper `runpy.run_path(script_path, run_name="__main__")` or call a function you export.
   - Prefer adding a small wrapper function `run_<element>_internal_script()` in `main.py` that checks script path and calls runpy.

How to add a new element (double-spike method)
1. Create a new script or extend `main.py` with solver functions:
   - If a dedicated script, follow the same placement as internal-standard scripts: `src/<Element>_Double_spike.py`.
2. Define expected data layout:
   - Input Excel sheet name(s) and required column names — document them in the script header.
3. Export results to `results/` and optionally return a minimal summary (CSV) for the launcher to display/save.

Recommended developer practices
- Keep element logic isolated in its own script/module to avoid merging lots of domain code into `main.py`.
- Use consistent input file naming: `<Element>_data.xlsx`.
- Use try/except with clear user-facing error messages.
- Add unit tests under `tests/`:
  - Test parsing of input files
  - Test solver functions (use small synthetic examples)
  - Test menu logic (mock input)
- Maintain a `requirements.txt` for reproducible environments.

Function hooks (suggested)
- run_<element>_internal_script() — wrapper to execute internal-standard script
- run_<element>_double_spike() — wrapper to execute double-spike solver (or return results)

Example: adding "Pb" to internal-standard
1. Add `src/Pb_Internal_standard_method.py` with required processing.
2. Add menu text and a wrapper in `src/main.py`:
   - Prompt: `2. Pb` (under Internal standard menu)
   - Call `run_pb_internal_script()` which runs the script.

---

## Troubleshooting & notes

- If scipy is missing, the program will terminate with an instruction to install it.
- Keep Excel inputs clean: no merged headers, consistent column names, and numeric columns parseable by `pandas.to_numeric`.
- Use encoding `utf-8-sig` when producing CSV for easy opening in Excel on Windows.

---

If you want, can also:
- Produce a `requirements.txt`.
- Add a template script for new elements (boilerplate).
- Add unit-test stubs under `tests/`.

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details.

## Author

Chufan Zhou (1176733817@qq.com)
