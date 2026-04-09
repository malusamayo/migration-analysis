import logging
import os


class GEPAFileLogger:
    """Logger that writes GEPA messages to a file and stderr."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        state_dir = os.path.join(run_dir, "shared", "state")
        os.makedirs(state_dir, exist_ok=True)
        self.log_path = os.path.join(state_dir, "gepa.log")

        self._logger = logging.getLogger(f"gepa.{id(self)}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        fh = logging.FileHandler(self.log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        self._logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(sh)

    def log(self, message: str):
        self._logger.info(message)
