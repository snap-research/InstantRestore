from typing import NamedTuple, List, Dict, Optional, Any

import torch
from PIL import Image


class BatchResults(NamedTuple):
    batch: Dict[str, Any]
    pred: torch.Tensor
    conditions: torch.Tensor
    loss: torch.Tensor
    loss_dict: Dict[str, float]
    pred_mix_id: torch.Tensor = None


class ProcessedResult(NamedTuple):
    image: Optional[Image.Image]
    scores: List[float]
    gt_ind: int
    loss: float
    path: Optional[str]
