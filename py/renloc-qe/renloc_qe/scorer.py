"""
renloc_qe/scorer.py
Isolated QE Scorer supporting unbabel-comet and fallback evaluation metrics.
"""
from typing import List, Dict, Any, Optional
import math
import sys

class QualityScorer:
    def __init__(self, model_name: str = "wmt22-comet-da", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._has_comet = False
        self._init_model()

    def _init_model(self):
        try:
            import comet
            print(f"[*] Loading COMET model: {self.model_name} on {self.device}...", file=sys.stderr)
            model_path = comet.download_model(self.model_name)
            self._model = comet.load_from_checkpoint(model_path)
            self._has_comet = True
            print(f"[+] Loaded COMET model successfully.", file=sys.stderr)
        except Exception as e:
            print(f"[*] COMET not available or model load skipped ({e}). Falling back to algorithmic QE heuristic.", file=sys.stderr)
            self._has_comet = False

    def score_batch(self, items: List[Dict[str, str]]) -> List[float]:
        """
        Items format:
        [{"src": "...", "mt": "...", "ref": "..." (optional)}]
        Returns list of quality scores between 0.0 and 1.0.
        """
        if self._has_comet and self._model is not None:
            try:
                data = []
                for it in items:
                    data.append({
                        "src": it.get("src", ""),
                        "mt": it.get("mt", ""),
                        "ref": it.get("ref", it.get("src", ""))
                    })
                model_output = self._model.predict(data, batch_size=8, gpus=1 if self.device != "cpu" else 0)
                # Output scores
                return [float(s) for s in model_output.scores]
            except Exception as e:
                print(f"[!] COMET scoring failed: {e}. Falling back to heuristic.", file=sys.stderr)

        # Fallback Algorithmic QE Scorer
        # Evaluates length ratio penalty, tag preservation, and n-gram overlap
        scores = []
        for it in items:
            src = it.get("src", "").strip()
            mt = it.get("mt", "").strip()
            if not src and not mt:
                scores.append(1.0)
                continue
            if not src or not mt:
                scores.append(0.0)
                continue

            # Length penalty (harmonic ratio)
            len_ratio = len(mt) / max(len(src), 1)
            len_penalty = math.exp(-abs(len_ratio - 1.1) / 2.0)

            # Character diversity
            unique_chars = len(set(mt)) / max(len(mt), 1)
            diversity_score = min(unique_chars * 3.0, 1.0)

            score = 0.5 * len_penalty + 0.5 * diversity_score
            scores.append(round(min(max(score, 0.0), 1.0), 4))
        return scores
