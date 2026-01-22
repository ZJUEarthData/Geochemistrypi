# GeochemistryPi — Chemical Modeling Module

The Chemical Modeling module provides a structured framework for geochemical computations with an extensible task-method-element architecture. It supports both interactive CLI and programmatic usage.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages: `pandas`, `numpy`, `scipy`, `openpyxl`, `xlsxwriter`

### Installation

```bash
# Clone the repository
git clone https://github.com/ZJUEarthData/Geochemistrypi.git
```
```bash
cd Geochemistrypi
```
```bash
# Create and activate virtual environment (recommended)
python3 -m venv .venv
```
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
```bash
# Install dependencies
pip install -r requirements.txt
```
```bash
# From project root
python -m geochemistrypi.cli

# Follow the prompts:
# 1. Choose option 1 for “data mining”, 2 for "chemical modeling"
# 2. Select input file or use sample data
# 3. Choose task → method → element
```

### Expected Output & Interaction:
```bash
Welcome to Geochemistry π
Please choose a module to run:
1 - Data Mining (automated ML pipelines)
2 - Chemical Modeling (equilibrium / fractionation / etc.)
Q - Quit
Enter 1 or 2 (or Q to quit): 2

Chemical Modeling launcher
Run non-interactively with a file? (y/N): n

Input Data File
Please provide the path to your input data file (Excel format).
You can:
1. Enter a file path
2. Press Enter to use sample data (if available)
3. Type 'exit' to cancel

File path (or press Enter for sample data):
```
### Using Sample Data

The module includes ready-to-use sample datasets. When prompted for file path, simply press Enter:
```bash
File path (or press Enter for sample data): [Press Enter]

[green]Using sample data: Hg_data.xlsx[/green]

Select task:
1. algo_fractionation
2. algo_equilibrium
3. algo_kinetic
4. algo_thermodynamic
5. algo_transport
Enter number (or blank to cancel): 1

Select method:
1. internal_standard: Internal standard method
2. double_spike: Double-spike (double-diluent) method
Enter number (or blank to cancel): 1

Select element:
1. Hg
Enter number (or blank to cancel): 1

Running -> task=algo_fractionation, method=internal_standard, element=Hg
Finished.
{'status': 'success', 'out_path': '/path/to/results/Hg_results.xlsx'}
```

## 📁 Project Structure
The structure refers to the data mining section

```bash
geochemistrypi/chemical_modeling/
├── __init__.py              # Module initialization
├── cli_pipeline.py          # Main CLI entry point
├── dispatcher.py            # Task/method/element discovery
├── data/
│   ├── data_readiness.py    # Data loading utilities
│   ├── Hg_data.xlsx         # Sample Hg data
│   └── Mo_data.xlsx         # Sample Mo data
├── model/func/
│   ├── algo_fractionation/  # Fractionation task implementations
│   │   ├── __init__.py
│   │   ├── internal_standard.py  # Internal standard method
│   │   └── double_spike.py       # Double-spike method
│   ├── algo_equilibrium/    # Equilibrium tasks (placeholder)
│   ├── algo_kinetic/        # Kinetic tasks (placeholder)
│   ├── algo_thermodynamic/  # Thermodynamic tasks (placeholder)
│   └── algo_transport/      # Transport tasks (placeholder)
├── process/
│   ├── hg_internal.py       # Hg-specific processing
│   └── mo_double_spike.py   # Mo-specific processing
└── results/                 # Output directory (gitignored)
```

## 🔧 Current Implementations

### Available Tasks

algo_fractionation: Isotope fractionation calculations
### Available Methods

1.Internal Standard Method

Elements: Hg
Implementation: process/hg_internal.py

2.Double-Spike Method

Elements: Mo
Implementation: process/mo_double_spike.py

## 🛠️ For Developers

Extending the Module

Adding a New Element to Existing Method

Create element-specific processing script in process/:

```bash
# process/new_element.py
def run(input_path: str, out_dir: str) -> dict:
    # Your implementation
    return {"status": "success", "output": "path/to/results"}
Register the element in the method's __init__.py:
```
```bash
# model/func/algo_fractionation/internal_standard.py
from ..process.new_element import run as new_element_run

def run(element: str, input_path: str, out_dir: str, **kwargs):
    if element == "NewElement":
        return new_element_run(input_path, out_dir, **kwargs)
Adding a New Method

Create method implementation in model/func/<task>/:
```
```bash
# model/func/algo_fractionation/new_method.py
def run(element: str, input_path: str, out_dir: str, **kwargs):
    # Dispatch to element-specific implementations
    pass
Register in task's __init__.py:
```
```bash
# model/func/algo_fractionation/__init__.py
from .new_method import run as new_method_run

def run(method: str, element: str, input_path: str, out_dir: str, **kwargs):
    if method == "new_method":
        return new_method_run(element, input_path, out_dir, **kwargs)
Adding a New Task

Create task directory structure:

text
model/func/new_task/
├── __init__.py
└── method1.py
Implement task dispatcher:
```
```bash
# model/func/new_task/__init__.py
def run(method: str, element: str, input_path: str, out_dir: str, **kwargs):
    # Task logic
    pass
```

### Development Practices

Code Style: Follow PEP 8, enforced via pre-commit (black, isort, flake8)
Testing: Add unit tests for new implementations
Documentation: Update docstrings and README
Data Validation: Use data/data_readiness.py utilities

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors & Contributors

Chufan Zhou (1176733817@qq.com) - Initial implementation
GeochemistryPi Development Team
