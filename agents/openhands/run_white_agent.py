import logging
import os
import re
import shlex
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import docker
import docker.errors
import tomli_w
from simple_parsing import ArgumentGenerationMode, ArgumentParser, flag

from rcabench.task.gen_task import prepare_task_asssets

ENVS = ["DOCKER_HOST"]
OPENAI_PREFIXES = ["gpt-", "o3", "o4"]
ANTHROPIC_PREFIXES = ["claude-"]

SCRIPT_DIR = Path(__file__).parent.absolute()
logger = logging.getLogger(__name__)


class OpenHandsError(Exception):
    """Base class for OpenHands errors"""
    pass


class OpenHandsTimeoutError(OpenHandsError):
    """Exception raised when OpenHands times out"""
    pass


@dataclass
class LLMArgs:
    model: str
    api_key: str | None = None
    base_url: str = ""
    top_p: float = 1.0
    temperature: float = 0.0
    max_output_tokens: int = 2048
    seed: int | None = None


@dataclass
class OpenhandsArgs:
    log_dir: Path
    tmp_dir: Path
    llm: LLMArgs
    max_iter: int = 30
    repo: Path = SCRIPT_DIR / "openhands-repo"
    silent: bool = False
    remove_tmp: bool = True
    timeout: int = 1200
    debug: bool = flag(default=False)


@dataclass
class TaskArgs:
    arvo_id: str
    """ARVO task ID (e.g., '10055')"""
    
    workspace_path: Path
    """Path to workspace directory"""
    
    cache_path: Path
    """Path to cache directory"""
    
    server: str
    """Server address for patch validation"""


def get_api_key(model: str):
    if any(model.startswith(prefix) for prefix in OPENAI_PREFIXES):
        env_var = "OPENAI_API_KEY"
    elif any(model.startswith(prefix) for prefix in ANTHROPIC_PREFIXES):
        env_var = "ANTHROPIC_API_KEY"
    else:
        env_var = "LLM_API_KEY"
    
    api_key = os.getenv(env_var)
    return api_key if api_key else "EMPTY"


def model_map(model: str):
    if model.startswith("claude-"):
        return model
    elif len(model.split("/")) >= 2:
        return model
    return f"openai/{model}"


def _cleanup_docker_container(log_dir: Path):
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        return
    
    pat = re.compile(r"runtime ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-[0-9a-f]{16})")
    with open(log_files[0]) as f:
        for line in f:
            match = pat.search(line)
            if match:
                container_id = match.group(1)
                break
        else:
            return
    
    client = docker.from_env()
    try:
        container = client.containers.get(f"openhands-runtime-{container_id}")
        container.remove(force=True)
        logger.info(f"Removed container {container_id}")
    except docker.errors.APIError as e:
        logger.warning(f"Container removal error: {e}")


def run_openhands(
    config_path: Path,
    prompt_path: Path,
    log_dir: Path,
    max_iter: int,
    timeout: int,
    model: str,
    llm_api_key: str | None = None,
    repo: Path = SCRIPT_DIR / "openhands-repo",
    silent: bool = False,
    debug: bool = False,
):
    # Use Python from the repo's virtual environment
    if os.name == 'nt':  # Windows
        python_path = Path(r"C:\Users\Asus\Downloads\aba\RCAbench\agents\openhands\rca312") / "Scripts" / "python.exe"
    else:  # Unix
        python_path = repo / ".venv" / "bin" / "python"
    
    if not python_path.exists():
        raise Exception(f"Python venv not found at {python_path}. Run 'poetry install' in {repo}")
    
    cmd = [
        str(python_path),
        "-m", "openhands.core.main",
        "--config-file", str(config_path.absolute()),
        "--file", str(prompt_path.absolute()),
        "--max-iterations", str(max_iter),
    ]
    
    env = os.environ.copy()  # Copy all env vars
    for env_var in ENVS:
        if os.getenv(env_var) is not None:
            env[env_var] = os.getenv(env_var)
    
    env["LLM_API_KEY"] = llm_api_key or get_api_key(model)
    env["LOG_TO_FILE"] = "1"
    env["LOG_DIR"] = str(log_dir.absolute())
    if debug:
        env["DEBUG"] = "1"
    env["LOG_ALL_EVENTS"] = "1"
    
    logger.info(f"Running OpenHands: {shlex.join(cmd)}")
    logger.info(f"Working directory: {repo}")
    try:
        subprocess.run(
            cmd,
            cwd=repo,
            env=env,
            stdout=subprocess.DEVNULL if silent else None,
            stderr=subprocess.DEVNULL if silent else None,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error("OpenHands timed out")
        raise OpenHandsTimeoutError("OpenHands timed out") from None
    finally:
        _cleanup_docker_container(log_dir)


def run_with_configs(openhands_args: OpenhandsArgs, task_args: TaskArgs):
    openhands_args.tmp_dir.mkdir(parents=True, exist_ok=True)
    openhands_args.log_dir.mkdir(parents=True, exist_ok=True)
    
    agent_id = uuid4().hex
    sub_dir = f"arvo_{task_args.arvo_id}-{agent_id}"
    tmp_input_dir = openhands_args.tmp_dir / sub_dir
    tmp_input_dir.mkdir()
    
    # 1. Copy template
    shutil.copytree(SCRIPT_DIR / "template", tmp_input_dir / "template")
    
    # 2. Prepare task assets
    task_dir = tmp_input_dir / "workspace"
    task_dir.mkdir()
    
    prepare_task_asssets(
        arvo_id=task_args.arvo_id,
        workspace_path=task_dir,
        cache_path=task_args.cache_path,
    )
    
    # 3. Create shared directory for submissions
    shared_dir = task_args.workspace_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Setup log directory
    log_dir = openhands_args.log_dir / sub_dir
    log_dir.mkdir()
    
    # 5. Configure OpenHands
    config_path = tmp_input_dir / "template" / "config.toml"
    with open(config_path) as f:
        config = tomllib.loads(f.read())
    
    config["core"]["workspace_base"] = str(task_dir.absolute()).replace('\\', '/')
    config["core"]["cache_dir"] = str((log_dir / "cache").absolute()).replace('\\', '/')
    config["core"]["file_store_path"] = str((log_dir / "file").absolute()).replace('\\', '/')
    config["core"]["save_trajectory_path"] = str((log_dir / "trajectory").absolute()).replace('\\', '/')
    config["llm"]["model"] = model_map(openhands_args.llm.model)
    config["llm"]["top_p"] = openhands_args.llm.top_p
    config["llm"]["temperature"] = openhands_args.llm.temperature
    config["llm"]["base_url"] = openhands_args.llm.base_url
    config["llm"]["max_output_tokens"] = openhands_args.llm.max_output_tokens
    
    if openhands_args.llm.seed is not None:
        config["llm"]["seed"] = openhands_args.llm.seed
    
    with open(config_path, "w") as f:
        f.write(tomli_w.dumps(config))
    
    # 6. Run OpenHands
    run_openhands(
        config_path=config_path,
        prompt_path=tmp_input_dir / "template" / "prompt.txt",
        log_dir=log_dir / "logs",
        timeout=openhands_args.timeout,
        repo=openhands_args.repo,
        silent=openhands_args.silent,
        max_iter=openhands_args.max_iter,
        model=openhands_args.llm.model,
        llm_api_key=openhands_args.llm.api_key,
        debug=openhands_args.debug,
    )
    
    # 7. Cleanup
    if openhands_args.remove_tmp:
        shutil.rmtree(tmp_input_dir, ignore_errors=True)
    
    return agent_id


def main(raw_args=None):
    parser = ArgumentParser(argument_generation_mode=ArgumentGenerationMode.BOTH)
    parser.add_arguments(OpenhandsArgs, dest="openhands_args")
    parser.add_arguments(TaskArgs, dest="task_args")
    
    args = parser.parse_args(raw_args)
    run_with_configs(args.openhands_args, args.task_args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()