"""YAML prompt library — loads prompts/*.yaml and renders by ``name.variant``.

Usage::

    lib = PromptLibrary()                    # loads all YAML files from prompts/
    text = lib.render("generic.simple")      # "Is there any defect…"
    text = lib.render("manufacturing.cot")

The library is a thin dict wrapper; all rendering is just a string lookup.
Both the Python toolkit and the iOS app consume the same YAML files so
cloud and edge runs use byte-identical prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Default location — overridable for tests.
_DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"


class PromptLibrary:
    """Loads all ``*.yaml`` files from a prompts directory.

    Args:
        prompts_dir: Directory containing ``*.yaml`` prompt files.
            Defaults to the repo-root ``prompts/`` directory resolved
            relative to this file.

    Raises:
        FileNotFoundError: If ``prompts_dir`` does not exist.
        ValueError: If a YAML file is missing the ``name`` or ``variants``
            key, or if a requested key does not exist.
    """

    def __init__(self, prompts_dir: Path | str | None = None) -> None:
        self._dir = Path(prompts_dir).resolve() if prompts_dir else _DEFAULT_PROMPTS_DIR.resolve()
        if not self._dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self._dir}")
        self._prompts: dict[str, dict[str, str]] = {}
        self._load_all()

    # ------------------------------------------------------------------

    def render(self, key: str) -> str:
        """Return the prompt text for ``key``.

        Args:
            key: ``"<name>.<variant>"`` e.g. ``"generic.simple"``.

        Returns:
            The prompt string with trailing whitespace stripped.

        Raises:
            ValueError: If the key does not exist in the library.
        """
        if "." not in key:
            raise ValueError(
                f"Prompt key must be '<name>.<variant>', got {key!r}. "
                f"Available: {self.available_keys()}"
            )
        name, variant = key.split(".", 1)
        if name not in self._prompts:
            raise ValueError(
                f"Prompt name {name!r} not found. Available names: {sorted(self._prompts)}"
            )
        variants = self._prompts[name]
        if variant not in variants:
            raise ValueError(
                f"Variant {variant!r} not found in {name!r}. Available variants: {sorted(variants)}"
            )
        return variants[variant].rstrip()

    def available_keys(self) -> list[str]:
        """Return all ``"<name>.<variant>"`` keys in the library."""
        keys: list[str] = []
        for name, variants in sorted(self._prompts.items()):
            for variant in sorted(variants):
                keys.append(f"{name}.{variant}")
        return keys

    def available_names(self) -> list[str]:
        """Return the top-level prompt names (one per YAML file)."""
        return sorted(self._prompts)

    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        for yaml_path in sorted(self._dir.glob("*.yaml")):
            self._load_file(yaml_path)

    def _load_file(self, path: Path) -> None:
        with path.open(encoding="utf-8") as f:
            data: Any = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping in {path.name}, got {type(data).__name__}")

        name = data.get("name")
        if not name:
            raise ValueError(f"Missing 'name' key in {path.name}")

        variants = data.get("variants")
        if not isinstance(variants, dict):
            raise ValueError(f"Missing or invalid 'variants' key in {path.name}")

        self._prompts[name] = {k: str(v) for k, v in variants.items()}
