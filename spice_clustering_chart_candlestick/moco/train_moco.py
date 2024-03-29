# CLI実行用
import sys
from rich import print
sys.path.append("/home/shohei/crypto/my_code/custom/spice_clustering_chart_candlestick/spice_clustering_chart_candlestick")

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
from rich import print

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    if cfg.general.seed is not None:
        pl.seed_everything(seed=cfg.general.seed, workers=True)

    # config.yamlで定義されたハイパーパラメーター一覧を取得
    cfg_hparams = flatten_omegaconf(cfg)

    cdm = ChartDataModule(
        cfg=cfg
    )

    model = build_model(
        cfg=cfg,
        cfg_hparams=cfg_hparams,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[callbacks.PrintCallback()],
    )

    trainer.fit(
        model=model,
        datamodule=cdm,
    )


if __name__ == "__main__":
    main()
