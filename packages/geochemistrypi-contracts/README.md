# GeochemistryPi Contracts

`geochemistrypi-contracts` contains the versioned JSON wire contracts shared by
the GeochemistryPi engine, runtime, and MCP environments.

The package is intentionally lightweight:

- it supports Python 3.9 and later;
- it has no runtime dependencies;
- it does not import pandas, scikit-learn, Pydantic, MCP, or the main
  GeochemistryPi package;
- it ships every public JSON Schema inside its wheel.

## Install for development

```text
python -m pip install -e packages/geochemistrypi-contracts
```

## Load a schema

```python
from geochemistrypi_contracts import SchemaName, load_schema, schema_sha256

schema = load_schema(SchemaName.CLASSIFICATION_EXPERIMENT_SPEC)
digest = schema_sha256(SchemaName.CLASSIFICATION_EXPERIMENT_SPEC)
```

The JSON Schema files are the normative cross-process contract. The standard
library dataclasses provide the engine-side representation and are kept aligned
with the schemas by round-trip tests.
