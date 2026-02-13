import os
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Dict

from app.core.config import resolve_repo_root


@dataclass(frozen=True)
class PromptPairTemplate:
    name: str
    system_path: Path
    user_path: Path
    system_template: Template
    user_template: Template


class PromptLoader:
    def __init__(
        self,
        prompt_dir: Path,
        pair_templates: Dict[str, PromptPairTemplate],
    ) -> None:
        self.prompt_dir = prompt_dir
        self._pair_templates = pair_templates

    @staticmethod
    def _resolve_prompt_dir() -> Path:
        env_dir = os.getenv("PROMPT_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists() and p.is_dir():
                return p
            raise FileNotFoundError(f"PROMPT_DIR not found or not a directory: {p}")

        repo_root = resolve_repo_root()
        candidates = [
            repo_root / "app" / "prompts" / "templates",
            repo_root / "prompts",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate

        raise FileNotFoundError(
            "Prompt templates directory not found. "
            "Create app/prompts/templates or set PROMPT_DIR."
        )

    @classmethod
    def load_default(cls) -> "PromptLoader":
        prompt_dir = cls._resolve_prompt_dir()

        pair_mapping = {
            "ask_clarification": (
                "ask_clarification.system.ja.txt",
                "ask_clarification.user.ja.txt",
            ),
            "generate_answer": (
                "generate_answer.system.ja.txt",
                "generate_answer.user.ja.txt",
            ),
            "standalone_question": (
                "standalone_question.system.ja.txt",
                "standalone_question.user.ja.txt",
            ),
        }
        pair_templates: Dict[str, PromptPairTemplate] = {}
        for name, (system_filename, user_filename) in pair_mapping.items():
            system_path = prompt_dir / system_filename
            user_path = prompt_dir / user_filename
            if not system_path.exists():
                raise FileNotFoundError(f"prompt file not found: {system_path}")
            if not user_path.exists():
                raise FileNotFoundError(f"prompt file not found: {user_path}")

            system_text = system_path.read_text(encoding="utf-8")
            user_text = user_path.read_text(encoding="utf-8")
            pair_templates[name] = PromptPairTemplate(
                name=name,
                system_path=system_path,
                user_path=user_path,
                system_template=Template(system_text),
                user_template=Template(user_text),
            )

        return cls(prompt_dir=prompt_dir, pair_templates=pair_templates)

    def render_pair(self, name: str, **kwargs: str) -> Dict[str, str]:
        if name not in self._pair_templates:
            raise KeyError(f"unknown pair prompt name: {name}")

        safe_kwargs = {k: str(v) for k, v in kwargs.items()}
        pair = self._pair_templates[name]
        return {
            "system": pair.system_template.substitute(**safe_kwargs),
            "user": pair.user_template.substitute(**safe_kwargs),
        }

    def available_pairs(self) -> Dict[str, Dict[str, str]]:
        return {
            k: {
                "system": v.system_path.name,
                "user": v.user_path.name,
            }
            for k, v in self._pair_templates.items()
        }
