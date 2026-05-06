import warnings

import hydra
import sklearn
from omegaconf import DictConfig

from src.logger import ExecutorLogger
from src.training.download_data import download_Home_Credit_Default_Risk
from src.training.evaluate import main as evaluate
from src.training.process_data import read_process_data
from src.training.train import train_model


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    warnings.filterwarnings("ignore")
    sklearn.set_config(transform_output="default")

    logger = ExecutorLogger("training")
    logger.info("Training started")

    download_Home_Credit_Default_Risk(cfg)
    read_process_data(cfg, logger)
    train_model(cfg, logger)
    evaluate(cfg, logger)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
