from typing import Literal
from pathlib import Path
import docker
from docker.errors import DockerException
import requests
from enum import IntEnum
from fastapi import HTTPException

DEFAULT_DOCKER_TIMEOUT = 30  # seconds for docker container to run
DEFAULT_CMD_TIMEOUT = 120  # seconds for command to run

class CustomExitCode(IntEnum):
    Timeout = 300

def run_arvo_container(
    arvo_id: str,
    mode: Literal["vul", "fix"],
    patch_dir: Path,
    docker_timeout: int = DEFAULT_DOCKER_TIMEOUT,
    cmd_timeout: int = DEFAULT_CMD_TIMEOUT,
):
    client = docker.from_env()
    container = None
    try:
        cmd = [
            "/bin/sh", "-c",
            f"patch -p1 < /tmp/patch/{arvo_id}_patch.diff;"
            "arvo compile &&"
            f"timeout {cmd_timeout} arvo"
        ]

        container = client.containers.run(
            image=f"n132/arvo:{arvo_id}-{mode}",
            command=cmd,
            volumes={
                str(patch_dir.resolve()): {"bind": "/tmp/patch", "mode": "ro"}},  # noqa: S108
            detach=True,
        )
        out = container.logs(stdout=True, stderr=False, stream=True, follow=True)
        exit_code = container.wait(timeout=docker_timeout)["StatusCode"]
        if exit_code == 137:  # Process killed by timeout
            exit_code = CustomExitCode.Timeout
            docker_output = b""
        else:
            docker_output = b"".join(out)
    except requests.exceptions.ReadTimeout:
        raise HTTPException(status_code=500, detail="Timeout waiting for the program") from None
    except DockerException as e:
        raise HTTPException(status_code=500, detail=f"Running error: {e}") from None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from None
    finally:
        if container:
            container.remove(force=True)

    return exit_code, docker_output