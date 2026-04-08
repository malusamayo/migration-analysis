"""Offline-first WebArena server lifecycle management using local Docker images."""
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

# Sites that can be started with a simple Docker container (no data volumes needed)
WEBARENA_BASIC_SITES = ["shopping", "shopping_admin", "reddit", "gitlab"]
WEBARENA_ALL_SITES = ["shopping", "shopping_admin", "reddit", "gitlab", "wikipedia", "map"]

# Default ports for webarena containers
DEFAULT_PORTS = {
    "shopping": 7770,
    "shopping_admin": 7780,
    "reddit": 9999,
    "gitlab": 8023,
    "wikipedia": 8888,
    "map": 3000,
}

CONTAINER_PORTS = {
    "shopping": 80,
    "shopping_admin": 80,
    "reddit": 80,
    "gitlab": 8023,
}

WEBARENA_IMAGES = {
    "shopping": "am1n3e/webarena-verified-shopping",
    "shopping_admin": "am1n3e/webarena-verified-shopping_admin",
    "reddit": "am1n3e/webarena-verified-reddit",
    "gitlab": "am1n3e/webarena-verified-gitlab",
}

# URL path suffixes per site
URL_SUFFIXES = {
    "shopping_admin": "/admin",
}

# Maps site names to WA_* environment variable names
WA_ENV_VARS = {
    "shopping": "WA_SHOPPING",
    "shopping_admin": "WA_SHOPPING_ADMIN",
    "reddit": "WA_REDDIT",
    "gitlab": "WA_GITLAB",
    "wikipedia": "WA_WIKIPEDIA",
    "map": "WA_MAP",
    "homepage": "WA_HOMEPAGE",
}

# Maps __PLACEHOLDER__ strings to WA_* env var names
URL_PLACEHOLDER_MAP = {
    "__REDDIT__": "WA_REDDIT",
    "__SHOPPING__": "WA_SHOPPING",
    "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",
    "__GITLAB__": "WA_GITLAB",
    "__WIKIPEDIA__": "WA_WIKIPEDIA",
    "__MAP__": "WA_MAP",
    "__HOMEPAGE__": "WA_HOMEPAGE",
}


WEBARENA_DOCKER_NETWORK = "webarena-net"


def _default_url(site: str) -> str:
    port = DEFAULT_PORTS[site]
    suffix = URL_SUFFIXES.get(site, "")
    return f"http://localhost:{port}{suffix}"


def _container_url(site: str) -> str:
    """URL for accessing a site from within the Docker network (by container name)."""
    name = _container_name(site)
    container_port = CONTAINER_PORTS[site]
    suffix = URL_SUFFIXES.get(site, "")
    if container_port == 80:
        return f"http://{name}{suffix}"
    return f"http://{name}:{container_port}{suffix}"


def _container_name(site: str) -> str:
    return f"webarena_verified_{site}"


def _env_ctrl_port(site: str) -> int:
    return DEFAULT_PORTS[site] + 1


def _run_docker(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _image_exists(site: str) -> bool:
    image = WEBARENA_IMAGES[site]
    result = _run_docker(["image", "inspect", image])
    return result.returncode == 0


def _container_exists(site: str) -> bool:
    result = _run_docker(["container", "inspect", _container_name(site)])
    return result.returncode == 0


def _container_running(site: str) -> bool:
    result = _run_docker(
        ["container", "inspect", "-f", "{{.State.Running}}", _container_name(site)]
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _wait_for_url(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}")


def _start_container(site: str) -> None:
    name = _container_name(site)
    if _container_exists(site):
        result = _run_docker(["start", name])
    else:
        image = WEBARENA_IMAGES[site]
        host_port = str(DEFAULT_PORTS[site])
        env_ctrl_port = str(_env_ctrl_port(site))
        container_port = str(CONTAINER_PORTS[site])
        result = _run_docker(
            [
                "run",
                "-d",
                "--name",
                name,
                "-p",
                f"{host_port}:{container_port}",
                "-p",
                f"{env_ctrl_port}:8877",
                "-e",
                f"WA_ENV_CTRL_EXTERNAL_SITE_URL=http://localhost:{host_port}",
                image,
            ]
        )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def _ensure_site_running(site: str, timeout: int) -> tuple[str, bool]:
    env_var = WA_ENV_VARS[site]
    existing_url = os.environ.get(env_var)
    if existing_url:
        _wait_for_url(existing_url, timeout=min(timeout, 30))
        return existing_url, False

    if site not in WEBARENA_IMAGES:
        raise RuntimeError(
            f"Offline startup is only implemented for {WEBARENA_BASIC_SITES}; got {site}."
        )

    if not _container_exists(site) and not _image_exists(site):
        raise RuntimeError(
            "Offline startup requires a local Docker image for "
            f"{site}: {WEBARENA_IMAGES[site]}.\n"
            "Pull/build it once while online, or set the corresponding WA_* "
            "environment variable and disable auto-start."
        )

    started_here = False
    if not _container_running(site):
        _start_container(site)
        started_here = True

    url = _default_url(site)
    _wait_for_url(url, timeout)
    os.environ[env_var] = url
    return url, started_here


def collect_required_sites(args_list: list[dict]) -> list[str]:
    sites: list[str] = []
    seen: set[str] = set()
    for args in args_list:
        sites_val = args.get("example", {}).get("sites", [])
        if isinstance(sites_val, list):
            site_list = sites_val
        else:
            site_list = [s.strip() for s in str(sites_val).split(",") if s.strip()]
        for site in site_list:
            if site and site not in seen:
                sites.append(site)
                seen.add(site)
    return sites


def set_default_webarena_urls(sites: Optional[list] = None) -> None:
    """Populate WA_* URLs with localhost defaults when callers manage servers externally."""
    if sites is None:
        sites = WEBARENA_BASIC_SITES

    for site in sites:
        env_var = WA_ENV_VARS.get(site)
        if env_var and not os.environ.get(env_var):
            os.environ[env_var] = _default_url(site)


def start_webarena_servers(
    sites: Optional[list] = None,
    timeout: int = 300,
) -> dict:
    """
    Start only the requested WebArena servers using local Docker state.

    Returns a mapping of sites started by this call to their URLs.
    """
    if sites is None:
        sites = WEBARENA_BASIC_SITES

    started_urls = {}
    errors = {}

    def _start_one(site: str) -> None:
        try:
            url, started_here = _ensure_site_running(site, timeout)
            print(f"  ✅ {site}: {url}")
            if started_here:
                started_urls[site] = url
        except Exception as e:
            errors[site] = str(e)

    print(f"🚀 Starting webarena servers offline: {sites}")
    threads = [threading.Thread(target=_start_one, args=(s,), daemon=True) for s in sites]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        raise RuntimeError(f"Failed to start webarena servers: {errors}")

    return started_urls


def stop_webarena_servers(sites: Optional[list] = None) -> None:
    """Stop WebArena Docker containers that were started for this run."""
    if sites is None:
        sites = WEBARENA_BASIC_SITES

    print(f"🛑 Stopping webarena servers: {sites}")
    for site in sites:
        if site not in WEBARENA_IMAGES:
            continue
        _run_docker(["stop", _container_name(site)])


def replace_url_placeholders(text: str) -> str:
    """Replace __SITE__ placeholders with actual URLs from WA_* env vars."""
    for placeholder, env_var in URL_PLACEHOLDER_MAP.items():
        url = os.environ.get(env_var, "")
        if url:
            text = text.replace(placeholder, url)
    return text


def preprocess_example(example: dict) -> dict:
    """Preprocess a webarena example: replace URL placeholders and append start URL to prompt."""
    example = example.copy()
    start_url = replace_url_placeholders(str(example.get("start_url", "")))
    example["prompt"] = f"{example['prompt']}\n\nStarting URL: {start_url}"
    return example


# Sites using Magento whose base URL must be reinitialized when accessed via Docker network.
# Magento embeds its base URL in redirects, so it must match the URL the agent will use.
MAGENTO_SITES = {"shopping", "shopping_admin"}


def _ensure_network_exists(network: str) -> None:
    result = _run_docker(["network", "inspect", network])
    if result.returncode != 0:
        _run_docker(["network", "create", network])


def _connect_to_network(site: str, network: str) -> None:
    """Connect a running container to the Docker network (no-op if already connected)."""
    _run_docker(["network", "connect", network, _container_name(site)])


def _get_container_ip(site: str, network: str) -> str:
    """Get the container's IP on the specified Docker network."""
    result = subprocess.run(
        ["docker", "inspect", _container_name(site),
         "--format", f'{{{{(index .NetworkSettings.Networks "{network}").IPAddress}}}}'],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _reinit_magento_base_url(site: str, base_url: str) -> None:
    """Run env-ctrl init inside the container to update Magento's base URL.

    Magento stores its base URL in the database. After connecting the container
    to a Docker network, the agent accesses it via a container IP, so Magento
    must be reconfigured to redirect to that IP instead of localhost.
    """
    print(f"  🔧 Reinitializing {site} base URL to {base_url}")
    subprocess.run(
        ["docker", "exec", _container_name(site), "env-ctrl", "init", "--base-url", base_url],
        capture_output=True, text=True, check=False, timeout=120,
    )


def set_container_webarena_urls(sites: list, network: str) -> None:
    """Create Docker network, connect running site containers, set WA_* to container IP URLs.

    For Magento-based sites (shopping, shopping_admin), also reinitializes the Magento
    base URL so that its internal redirects point to the container's IP on the network
    rather than localhost. This is required because Magento embeds the base URL in all
    redirects, and localhost is not resolvable from inside the agent's Docker container.

    Call this instead of set_default_webarena_urls when the agent runs in Docker.
    """
    _ensure_network_exists(network)
    for site in sites:
        if site not in WEBARENA_IMAGES:
            continue
        if not _container_running(site):
            continue
        _connect_to_network(site, network)
        env_var = WA_ENV_VARS.get(site)
        if not env_var:
            continue
        if site in MAGENTO_SITES:
            ip = _get_container_ip(site, network)
            if ip:
                base_url = f"http://{ip}/"
                _reinit_magento_base_url(site, base_url)
                suffix = URL_SUFFIXES.get(site, "")
                os.environ[env_var] = f"http://{ip}{suffix}"
            else:
                os.environ[env_var] = _container_url(site)
        else:
            os.environ[env_var] = _container_url(site)
