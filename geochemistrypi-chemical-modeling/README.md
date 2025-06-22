# GeochemistryPi Chemical Modeling

A Python package for solving geochemistry equations using different numerical methods (fsolve and root).

## Features

- Solve chemical equations using scipy.optimize.fsolve
- Solve chemical equations using scipy.optimize.root
- Compare results from both methods
- Command-line interface for easy use

## Installation

```bash
pip install geochemistrypi-chemical-modeling
```

## Usage

```bash
# Using fsolve method
geochemistrypi solve --initial-guess 0.5 0.5 2.0

# Using root method
geochemistrypi root-method --initial-guess 0.5 0.5 2.0

# Compare both methods
geochemistrypi compare --initial-guess 0.5 0.5 2.0
```

## Requirements

- Python >= 3.8
- numpy
- scipy
- typer
- rich

## Development

To contribute to this project:

1. Clone the repository
2. Install development dependencies:
```bash
pip install -e ".[dev]"
```
3. Run tests:
```bash
python -m pytest
```

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details.

## Author

Chufan Zhou (1176733817@qq.com)