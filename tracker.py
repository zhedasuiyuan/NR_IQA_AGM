"""
Unified experiment tracker.

Aim is a free, open-source, self-hosted alternative to Weights & Biases
(https://aimstack.readthedocs.io/) and is the default backend here. W&B is
still selectable for backward compatibility, and "none" disables tracking.

Only the small surface used by train.py is wrapped: scalar metric logging,
hyper-parameter recording, an optional source-code snapshot, and teardown.
Backend libraries are imported lazily so that installing only the chosen
backend is enough.
"""

from __future__ import annotations

from typing import Optional


class Tracker:
    """Thin wrapper over an experiment-tracking backend.

    Parameters
    ----------
    backend : "aim" | "wandb" | "none"
        Tracking backend. "none" turns logging off entirely.
    project : str
        Project / experiment name shown in the UI.
    name : str, optional
        Human-readable run name.
    config : dict, optional
        Hyper-parameters recorded with the run.
    enabled : bool
        Set False on non-main processes so only rank-0 logs.
    """

    def __init__(
        self,
        backend: str = "aim",
        *,
        project: str = "NR_IQA_AGM",
        name: Optional[str] = None,
        config: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.backend = backend
        self.enabled = enabled and backend != "none"
        self._run = None
        if not self.enabled:
            return

        if backend == "aim":
            from aim import Run

            self._run = Run(experiment=project, log_system_params=True)
            if name:
                self._run.name = name
            if config:
                self._run["hparams"] = config
        elif backend == "wandb":
            import wandb

            self._run = wandb.init(project=project, name=name, config=config)
        else:
            raise ValueError(f"Unknown tracker backend: {backend!r}")

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log a dict of scalar metrics, optionally at an explicit step."""
        if not self.enabled:
            return
        if self.backend == "aim":
            for key, value in metrics.items():
                self._run.track(value, name=key, step=step)
        else:  # wandb
            import wandb

            wandb.log(metrics, step=step)

    def log_code(self, path: str) -> None:
        """Snapshot a source file with the run (best-effort)."""
        if not self.enabled:
            return
        if self.backend == "aim":
            # Aim records git state and system info via log_system_params; it
            # has no lightweight first-class code artifact, so this is a no-op.
            return
        import wandb

        art = wandb.Artifact("source-code", type="code")
        art.add_file(path)
        wandb.log_artifact(art)

    def finish(self) -> None:
        """Close the run and flush any buffered data."""
        if not self.enabled or self._run is None:
            return
        if self.backend == "aim":
            self._run.close()
        else:
            import wandb

            wandb.finish()
