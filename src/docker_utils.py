import os
import platform
from typing import Optional

from dotenv import dotenv_values


def detect_platform() -> str:
    """Detect the correct platform string for Docker container images."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


def get_forward_env() -> list[str]:
    """Build the list of env vars to forward into Docker from .env and os.environ.

    Also forwards USER so getpass.getuser() works inside containers running as
    a non-/etc/passwd UID.
    """
    all_envs = os.environ.copy()
    env_config = dotenv_values(".env")
    forward_env = [key for key in env_config.keys() if key in all_envs]
    if "USER" in all_envs and "USER" not in forward_env:
        forward_env.append("USER")
    return forward_env


def make_docker_kwargs(
    working_dir: str,
    server_image: str,
    volumes: list[str],
    docker_network: Optional[str] = None,
) -> dict:
    """Build complete kwargs dict for DockerWorkspace.

    Appends the live SDK mount when the local checkout exists, and fills in
    platform, forward_env, and user automatically.
    """
    volumes = list(volumes)

    kwargs: dict = dict(
        working_dir=working_dir,
        server_image=server_image,
        platform=detect_platform(),
        volumes=volumes,
        forward_env=get_forward_env(),
        user=f"{os.getuid()}:{os.getgid()}",
    )
    if docker_network:
        kwargs["network"] = docker_network
    return kwargs
