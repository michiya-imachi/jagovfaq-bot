import sys
from pathlib import Path
from typing import Any, Optional, Tuple


def export_graph_artifacts(app: Any, out_dir: Path) -> Tuple[Path, Optional[Path]]:
    # Export Mermaid (.mmd) and PNG (.png) for the compiled graph.
    out_dir.mkdir(parents=True, exist_ok=True)

    g = app.get_graph()

    mermaid_text = g.draw_mermaid()
    mmd_path = out_dir / "graph.mmd"
    mmd_path.write_text(str(mermaid_text).rstrip() + "\n", encoding="utf-8")

    png_path = out_dir / "graph.png"
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
        print(
            f"WARNING: graph PNG export failed: {type(e).__name__}: {e}",
            file=sys.stderr,
            flush=True,
        )

    return mmd_path, (png_path if png_ok else None)
