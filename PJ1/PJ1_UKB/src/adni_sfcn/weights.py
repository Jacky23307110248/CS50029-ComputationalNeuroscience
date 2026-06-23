"""Load UKB age SFCN weights into ADNI 3-class classifier."""

from __future__ import annotations

import logging
from pathlib import Path

from ..models.sfcn_weights import load_raw_state_dict, resolve_sfcn_weight_file
from .classifier import SFCNClassifier, build_sfcn_classifier

logger = logging.getLogger(__name__)


def load_pretrained_sfcn_classifier(
    checkpoint_path: Path | str | None = None,
    num_classes: int = 3,
    pretrained: bool = True,
) -> SFCNClassifier:
    model = build_sfcn_classifier(num_classes=num_classes)
    if not pretrained:
        return model

    path = resolve_sfcn_weight_file(
        str(checkpoint_path) if checkpoint_path else "age_best"
    )
    state = load_raw_state_dict(path)
    fe_state = {
        (k[len("feature_extractor.") :] if k.startswith("feature_extractor.") else k): v
        for k, v in state.items()
        if k.startswith("feature_extractor.")
    }
    model.feature_extractor.load_state_dict(fe_state, strict=True)
    logger.info("Loaded SFCN age backbone from %s (3-class head random init)", path)
    return model
