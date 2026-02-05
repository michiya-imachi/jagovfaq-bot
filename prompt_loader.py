import os
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Dict


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    path: Path
    template: Template


class PromptLoader:
    def __init__(self, prompt_dir: Path, templates: Dict[str, PromptTemplate]) -> None:
        self.prompt_dir = prompt_dir
        self._templates = templates

    @staticmethod
    def _resolve_prompt_dir() -> Path:
        # Prefer explicit path via env var
        env_dir = os.getenv("PROMPT_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists() and p.is_dir():
                return p
            raise FileNotFoundError(f"PROMPT_DIR not found or not a directory: {p}")

        # Search upward from this file to find repo-root prompts/
        start = Path(__file__).resolve().parent
        for d in [start, *start.parents]:
            candidate = d / "prompts"
            if candidate.exists() and candidate.is_dir():
                return candidate

        raise FileNotFoundError(
            "prompts/ directory not found. Create prompts/ at repo root or set PROMPT_DIR."
        )

    @classmethod
    def load_default(cls) -> "PromptLoader":
        prompt_dir = cls._resolve_prompt_dir()

        mapping = {
            "ask_clarification": "ask_clarification.ja.txt",
            "generate_answer": "generate_answer.ja.txt",
        }

        templates: Dict[str, PromptTemplate] = {}
        for name, filename in mapping.items():
            path = prompt_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"prompt file not found: {path}")
            text = path.read_text(encoding="utf-8")
            templates[name] = PromptTemplate(
                name=name, path=path, template=Template(text)
            )

        return cls(prompt_dir=prompt_dir, templates=templates)

    def render(self, name: str, **kwargs: str) -> str:
        if name not in self._templates:
            raise KeyError(f"unknown prompt name: {name}")

        safe_kwargs = {k: str(v) for k, v in kwargs.items()}
        # Use substitute() to fail fast if variables are missing.
        return self._templates[name].template.substitute(**safe_kwargs)

    def available(self) -> Dict[str, str]:
        # Return prompt name -> filename for debugging/logging.
        return {k: v.path.name for k, v in self._templates.items()}
