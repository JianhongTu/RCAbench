# Global constants for RCAbench package

from pathlib import Path
from dataclasses import dataclass

# Default directories (as Path objects for type safety and consistency)

DEFAULT_DATA_DIR = Path("./data")  # Directory for task metadata and database
DEFAULT_CACHE_DIR = Path("./cache")  # Directory for cached files
DEFAULT_TEMP_DIR = Path("./tmp")  # Directory for temporary agent-specific workspaces


@dataclass
class AgentWorldPath:
    """Class to encapsulate paths used by an agent in RCAbench."""

    temp: Path = DEFAULT_TEMP_DIR
    arvo_id: int = 0
    agent_id: str = "agentid123"

    @property
    def base_dir(self) -> Path:
        return self.temp

    @property
    def agent_dir(self) -> Path:
        return self.temp / f"arvo_{self.arvo_id}-{self.agent_id}"

    @property
    def workspace_dir(self) -> Path:
        return self.agent_dir / "workspace"

    @property
    def shared_dir(self) -> Path:
        return self.workspace_dir / "shared"

    @property
    def codebase_dir(self) -> Path:
        return self.workspace_dir / "src-vul"

    @property
    def arvo_id_error_path(self) -> Path:
        return self.workspace_dir / f"{self.arvo_id}_error.txt"

    @property
    def codebase_compressed_path(self) -> Path:
        return self.workspace_dir / f"{self.arvo_id}_{CODEBASE_FILE_NAME}"

    @property
    def submit_loc_script_path(self) -> Path:
        return self.workspace_dir / "submit_loc.sh"

    @property
    def submit_patch_script_path(self) -> Path:
        return self.workspace_dir / "submit_patch.sh"


# Codebase file name on the cybergym HF repo
CODEBASE_FILE_NAME = "repo-vul.tar.gz"

# Name of the extracted source directory
CODEBASE_SRC_NAME = "src-vul"

# Expected directory structure:
# ./
# ├── data/                   # Task metadata (DEFAULT_DATA_DIR)
# │   ├── arvo.db            # SQLite database of Arvo tasks
# │   └── verified_jobs.json # List of verified task IDs
# ├── workspace/
# │   └── shared/            # Agent-server communication (DEFAULT_WORKSPACE_DIR)
# │       ├── loc.json       # Localization submissions
# │       └── patch.diff     # Patch submissions
# ├── cache/                 # Cached files (DEFAULT_CACHE_DIR)
# └── tmp/                   # Temporary agent workspaces (DEFAULT_TEMP_DIR)
#     └── arvo_{arvo_id}-{agent_id}/  # Agent-specific directory (multiple per ARVO task)
#         └── workspace/              # Working directory for the agent
#             ├── shared/             # Shared resources directory
#             ├── src-vul/            # Extracted vulnerable source code (CODEBASE_SRC_NAME)
#             ├── {arvo_id}_error.txt # Fuzzer error report
#             ├── submit_patch.sh     # Script to submit patches
#             └── submit_loc.sh       # Script to submit localization results

# Default server settings
DEFAULT_HOST_IP = "localhost"
DEFAULT_HOST_PORT = 8000
