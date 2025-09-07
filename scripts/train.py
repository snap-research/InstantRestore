import torch
import pyrallis
import cv2

import sys
sys.path.append(".")
sys.path.append("..")

from face_replace.training.coach import Coach
from face_replace.configs.train_config import TrainConfig

cv2.setNumThreads(0)
torch.autograd.set_detect_anomaly(True)


@pyrallis.wrap()
def main(cfg: TrainConfig):
    coach = Coach(cfg)
    coach.train()


if __name__ == '__main__':
    main()
