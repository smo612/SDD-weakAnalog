from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parent.parent
CAT_DIR = REPO / "data" / "cat"
KODAK_DIR = REPO / "data" / "kodak"
OUT_ROOT = REPO / "results_cpack_match_0506"
RESIZE_HW = (128, 128)
BASELINE_LOCAL = KODAK_DIR / "kodim01.png"
BASELINE_REMOTE = KODAK_DIR / "kodim24.png"


@dataclass
class MatchRow:
    remote_name: str
    selected_local: str
    baseline_similarity: float
    selected_similarity: float
    similarity_gap: float
    top3: list[dict]


def load_feature(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(RESIZE_HW, Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.reshape(-1, 3)
    mean = arr.mean(axis=0, keepdims=True)
    arr = arr - mean
    norm = np.linalg.norm(arr)
    if norm <= 1e-12:
        return arr.reshape(-1)
    return (arr / norm).reshape(-1)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def all_cat_images() -> list[Path]:
    return sorted([p for p in CAT_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def all_kodak_images() -> list[Path]:
    return sorted([p for p in KODAK_DIR.iterdir() if p.is_file() and p.name.lower().startswith("kodim") and p.suffix.lower() == ".png"])


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)

    baseline_local_feat = load_feature(BASELINE_LOCAL)
    baseline_remote_feat = load_feature(BASELINE_REMOTE)
    baseline_similarity = cosine_sim(baseline_local_feat, baseline_remote_feat)

    kodak_paths = all_kodak_images()
    kodak_features = {p.name: load_feature(p) for p in kodak_paths}

    rows: list[MatchRow] = []
    for cat_path in all_cat_images():
        cat_feat = load_feature(cat_path)
        scored = []
        for kodak_path in kodak_paths:
            sim = cosine_sim(kodak_features[kodak_path.name], cat_feat)
            scored.append(
                {
                    "local": kodak_path.name,
                    "similarity": sim,
                    "gap_to_baseline": abs(sim - baseline_similarity),
                }
            )
        scored.sort(key=lambda item: (item["gap_to_baseline"], -item["similarity"], item["local"]))
        best = scored[0]
        rows.append(
            MatchRow(
                remote_name=cat_path.name,
                selected_local=best["local"],
                baseline_similarity=baseline_similarity,
                selected_similarity=best["similarity"],
                similarity_gap=best["gap_to_baseline"],
                top3=scored[:3],
            )
        )

    payload = {
        "metric": "mean-centered RGB cosine similarity at resize_128",
        "baseline_pair": {
            "local": BASELINE_LOCAL.name,
            "remote": BASELINE_REMOTE.name,
            "similarity": baseline_similarity,
        },
        "matches": [
            {
                "remote_name": row.remote_name,
                "selected_local": row.selected_local,
                "baseline_similarity": row.baseline_similarity,
                "selected_similarity": row.selected_similarity,
                "similarity_gap": row.similarity_gap,
                "top3": row.top3,
            }
            for row in rows
        ],
    }

    (OUT_ROOT / "cat_local_matches.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Cat Local Matching",
        "",
        f"- Metric: `{payload['metric']}`",
        f"- Baseline pair: `{BASELINE_LOCAL.name}` -> `{BASELINE_REMOTE.name}`",
        f"- Baseline similarity: `{baseline_similarity:.6f}`",
        "",
        "| remote cat | selected local | selected similarity | gap to baseline | top-3 locals |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        top3 = ", ".join(f"{item['local']} ({item['similarity']:.4f})" for item in row.top3)
        md.append(
            f"| {row.remote_name} | {row.selected_local} | {row.selected_similarity:.6f} | {row.similarity_gap:.6f} | {top3} |"
        )
    (OUT_ROOT / "cat_local_matches.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[OK] wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
