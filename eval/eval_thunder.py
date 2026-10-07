"""
Evaluate OMNI encoders on the THUNDER benchmark (https://github.com/MICS-Lab/thunder).

Setup (once):
    pip install thunder-bench
    export THUNDER_BASE_DATA_FOLDER=/path/to/thunder_data   # empty folder, THUNDER fills it

The requested datasets are downloaded and their splits generated automatically
(skip this with --no-download once they are in place).

Usage (Python API):
    python eval_thunder.py --model base --datasets bach mhist --tasks knn linear_probing
    python eval_thunder.py --model large --datasets pannuke --tasks segmentation
    python eval_thunder.py --model sofieneb/omni-small --datasets bach --tasks knn simple_shot

Usage (THUNDER CLI, which instantiates the class below with its defaults):
    OMNI_MODEL=base thunder benchmark custom:eval_thunder.py bach knn
    OMNI_MODEL=base thunder benchmark custom:eval_thunder.py bach linear_probing --loading-mode=embedding_pre_loading

Results are written to:
    $THUNDER_BASE_DATA_FOLDER/outputs/res/<dataset>/<model>/<task>/<adaptation_type>/outputs.json
and can be summarised with:
    thunder results-summary
"""

import argparse
import os

import torch
from thunder import benchmark, download_datasets, generate_splits
from thunder.models import PretrainedModel
from transformers import AutoModel

# Short names -> Hugging Face repos
OMNI_MODELS = {
    "tiny": "sofieneb/omni-tiny",
    "small": "sofieneb/omni-small",
    "base": "sofieneb/omni-base",
    "large": "sofieneb/omni-large",
}
DEFAULT_MODEL = "base"

# Tasks that train on pre-computed embeddings (see the THUNDER examples)
PRELOADED_TASKS = {"linear_probing", "segmentation", "simple_shot"}


def resolve_model(model: str) -> str:
    """Map 'tiny' / 'small' / 'base' / 'large' to a repo id; pass anything else through (repo id or local folder)."""
    return OMNI_MODELS.get(model, model)


class OmniThunder(PretrainedModel):
    """
    THUNDER wrapper around an OMNI encoder.

    Linear probing / kNN / SimpleShot use the average of the teacher tokens;
    segmentation uses the patch tokens.

    Args:
        model (str): 'tiny', 'small', 'base', 'large', a Hugging Face repo id or a local folder.
            Defaults to the OMNI_MODEL environment variable, then to 'base'.
    """

    def __init__(self, model: str | None = None):
        super().__init__()
        repo_id = resolve_model(model or os.environ.get("OMNI_MODEL", DEFAULT_MODEL))
        self.model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()
        self.name = repo_id.rstrip("/").split("/")[-1]  # used by THUNDER to name output folders
        self.emb_dim = self.model.config.embed_dim
        self.vlm = False
        self._transform = self.model.get_transform()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Images of shape (B, 3, 224, 224).

        Returns:
            torch.Tensor: Tile embeddings of shape (B, D).
        """
        return self.model(x)

    def get_transform(self):
        """Returns the image preprocessing used by OMNI."""
        return self._transform

    def get_linear_probing_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Images of shape (B, 3, 224, 224).

        Returns:
            torch.Tensor: Average of the teacher tokens, shape (B, D).
        """
        return self.model(x)

    def get_segmentation_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Images of shape (B, 3, 224, 224).

        Returns:
            torch.Tensor: Patch tokens of shape (B, 256, D).
        """
        return self.model(x, return_patch_tokens=True)["patch_tokens"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate OMNI encoders on THUNDER.")
    parser.add_argument(
        "--model",
        default=os.environ.get("OMNI_MODEL", DEFAULT_MODEL),
        help="tiny, small, base, large, a Hugging Face repo id or a local folder (default: base).",
    )
    parser.add_argument("--datasets", nargs="+", default=["bach"], help="THUNDER dataset names.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["knn", "linear_probing"],
        help="THUNDER tasks: knn, linear_probing, simple_shot, segmentation, "
        "transformation_invariance, adversarial_attack.",
    )
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Skip downloading the datasets and generating their splits (done by default).",
    )
    parser.add_argument(
        "--no-recompute",
        action="store_true",
        help="Re-use embeddings already computed by a previous run.",
    )
    args = parser.parse_args()

    if not os.environ.get("THUNDER_BASE_DATA_FOLDER"):
        raise SystemExit("Set THUNDER_BASE_DATA_FOLDER first, e.g. export THUNDER_BASE_DATA_FOLDER=/path/to/thunder_data")

    if args.download:
        download_datasets(args.datasets)
        generate_splits(args.datasets)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OmniThunder(args.model).to(device).eval()
    print(f"Model: {model.name} (dim {model.emb_dim}) on {device}")

    for dataset in args.datasets:
        # Embeddings are computed once, then re-used by every task.
        benchmark(model, dataset=dataset, task="pre_computing_embeddings", recomp_embs=not args.no_recompute)
        for task in args.tasks:
            extra = {"loading_mode": "embedding_pre_loading"} if task in PRELOADED_TASKS else {}
            benchmark(model, dataset=dataset, task=task, retrain_model=True, **extra)


if __name__ == "__main__":
    main()