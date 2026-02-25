import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportGraphArtifactsResult:
    collapsed_mmd: Path
    collapsed_png: Optional[Path]
    expanded_mmd: Path
    expanded_png: Optional[Path]


def _export_single_graph(
    app: Any,
    out_dir: Path,
    *,
    xray: bool,
    base_name: str,
    label: str,
) -> tuple[Path, Optional[Path]]:
    # Export Mermaid (.mmd) and PNG (.png) for the compiled graph.
    g = app.get_graph(xray=xray)

    mermaid_text = g.draw_mermaid()
    mmd_path = out_dir / f"{base_name}.mmd"
    mmd_path.write_text(str(mermaid_text).rstrip() + "\n", encoding="utf-8")

    png_path = out_dir / f"{base_name}.png"
    png_ok = True

    try:
        # Some versions support saving directly via output_file_path.
        g.draw_mermaid_png(output_file_path=str(png_path))
    except TypeError:
        # Other versions return PNG bytes.
        png_bytes = g.draw_mermaid_png()
        if isinstance(png_bytes, (bytes, bytearray)):
            png_path.write_bytes(png_bytes)
        else:
            png_path.write_bytes(bytes(png_bytes))
    except Exception as e:
        png_ok = False
        logger.warning(
            "graph PNG export failed (%s): %s: %s",
            label,
            type(e).__name__,
            e,
        )

    return mmd_path, (png_path if png_ok else None)


def export_graph_artifacts(
    app: Any,
    out_dir: Path,
) -> ExportGraphArtifactsResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    collapsed_mmd, collapsed_png = _export_single_graph(
        app=app,
        out_dir=out_dir,
        xray=False,
        base_name="graph_collapsed",
        label="collapsed",
    )
    expanded_mmd, expanded_png = _export_single_graph(
        app=app,
        out_dir=out_dir,
        xray=True,
        base_name="graph_expanded",
        label="expanded",
    )

    return ExportGraphArtifactsResult(
        collapsed_mmd=collapsed_mmd,
        collapsed_png=collapsed_png,
        expanded_mmd=expanded_mmd,
        expanded_png=expanded_png,
    )
