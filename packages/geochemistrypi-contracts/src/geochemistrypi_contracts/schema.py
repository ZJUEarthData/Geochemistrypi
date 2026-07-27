"""Access packaged JSON Schemas without filesystem assumptions."""

import hashlib
import json
from enum import Enum
from importlib import resources
from typing import Any, Dict, Union

CONTRACT_VERSION = "1.0"
SCHEMA_BASE_URI = "https://schemas.geochemistrypi.org/contracts/v1/"
_SCHEMA_PACKAGE = "geochemistrypi_contracts.schemas.v1"


class SchemaName(str, Enum):
    """Stable names for public v1 contract schemas."""

    DATASET_REF = "dataset-ref"
    CLASSIFICATION_EXPERIMENT_SPEC = "classification-experiment-spec"
    EXPERIMENT_RESULT = "experiment-result"
    ERROR_RESPONSE = "error-response"


SCHEMA_FILENAMES = {
    SchemaName.DATASET_REF: "dataset-ref.schema.json",
    SchemaName.CLASSIFICATION_EXPERIMENT_SPEC: "classification-experiment-spec.schema.json",
    SchemaName.EXPERIMENT_RESULT: "experiment-result.schema.json",
    SchemaName.ERROR_RESPONSE: "error-response.schema.json",
}


def _schema_name(name: Union[SchemaName, str]) -> SchemaName:
    try:
        return name if isinstance(name, SchemaName) else SchemaName(name)
    except ValueError as exc:
        valid = ", ".join(item.value for item in SchemaName)
        raise ValueError(f"Unknown schema {name!r}. Expected one of: {valid}.") from exc


def schema_bytes(name: Union[SchemaName, str]) -> bytes:
    """Return the exact UTF-8 bytes shipped in the installed wheel."""

    filename = SCHEMA_FILENAMES[_schema_name(name)]
    return resources.files(_SCHEMA_PACKAGE).joinpath(filename).read_bytes()


def load_schema(name: Union[SchemaName, str]) -> Dict[str, Any]:
    """Load a packaged schema as a new dictionary."""

    return json.loads(schema_bytes(name).decode("utf-8"))


def schema_id(name: Union[SchemaName, str]) -> str:
    """Return the stable identifier declared by a packaged schema."""

    value = load_schema(name).get("$id")
    if not isinstance(value, str):
        raise ValueError(f"Schema {_schema_name(name).value!r} does not declare a string $id.")
    return value


def schema_sha256(name: Union[SchemaName, str]) -> str:
    """Hash the exact packaged schema bytes for provenance records."""

    return hashlib.sha256(schema_bytes(name)).hexdigest()
