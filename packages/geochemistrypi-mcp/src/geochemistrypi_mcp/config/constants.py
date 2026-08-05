"""Versioned compatibility constants for the wrapper boundary."""

SERVER_NAME = "GeochemistryPi MCP"
SERVER_VERSION = "0.2.0"
SUPPORTED_CLI_VERSIONS = ("0.8.0",)
INTERACTION_PLAN_VERSION = 1
CLI_AUTOMATION_CONTRACT_VERSION = 1
COMPATIBILITY_POLICY_VERSION = 2
MCP_PYTHON_REQUIRES = ">=3.10,<4"
CLI_PYTHON_REQUIRES = ">=3.9,<3.10"
MCP_SDK_REQUIRES = "==2.0.0"
ARTIFACT_INDEX_SCHEMA_VERSION = 1
TARGET_OPERATING_SYSTEMS = ("windows", "linux", "macos")
PUBLIC_RELEASE_READY = False
PENDING_RELEASE_GATES = (
    "full_pr9i_matrix_on_required_platforms",
    "clean_linux_and_macos_setup_acceptance",
    "real_client_natural_language_acceptance",
    "upgrade_from_last_published_bundle",
    "no_medium_or_higher_parity_defects",
    "signed_release_manifest_and_hashes",
    "pypi_and_mcp_registry_publication_authorization",
)

# Variables owned by the wrapper interpreter must never leak into the separate
# GeochemistryPi CLI interpreter. This matters especially for uv/uvx launches.
ISOLATED_CLI_ENVIRONMENT_VARIABLES = (
    "PYTHONHOME",
    "PYTHONPATH",
    "UV_INTERNAL__PYTHONHOME",
    "VIRTUAL_ENV",
    "__PYVENV_LAUNCHER__",
    "SQLALCHEMY_DATABASE_URL",
    "GEOCHEMISTRYPI_MCP_APP_ROOT",
    "GEOCHEMISTRYPI_CLI_EXECUTABLE",
    "GEOCHEMISTRYPI_MCP_RUNS_ROOT",
    "GEOCHEMISTRYPI_MCP_TRACKING_ROOT",
    "GEOCHEMISTRYPI_MCP_SERVICE_STATE_ROOT",
    "GEOCHEMISTRYPI_MCP_SETTINGS_FILE",
    "GEOCHEMISTRYPI_MCP_MAX_DATASET_BYTES",
    "GEOCHEMISTRYPI_MCP_MAX_PENDING_RUNS",
    "GEOCHEMISTRYPI_MCP_MAX_PROCESS_SECONDS",
)
