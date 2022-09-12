# CLI実行用
# import sys
# from rich import print
# sys.path.append("/home/shohei/crypto/my_code/custom/spice_clustering_chart_candlestick/spice_clustering_chart_candlestick")

# PyTorch Lightning
import pytorch_lightning as pl

# MoCo
import moco.callbacks as callbacks
from moco.build_model import build_model
from moco.data_module import ChartDataModule

# Utils
from utils.utils import flatten_omegaconf

# general
import pretty_errors
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.general.seed is not None:
        pl.seed_everything(seed=cfg.general.seed, workers=True)

    # config.yamlで定義されたハイパーパラメーター一覧を取得
    cfg_hparams = flatten_omegaconf(cfg)

    cdm = ChartDataModule(
        cfg.data
    )

    model = build_model(
        cfg,
        cfg_hparams,
    )

    trainer = pl.Trainer(
        callbacks=[callbacks.PrintCallback()],
        **cfg.trainer,
    )

    trainer.fit(
        model=model,
        datamodule=cdm,
    )


if __name__ == "__main__":
    main()
