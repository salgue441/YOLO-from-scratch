from models.yolo import YOLO
from utils.loss import YOLOLoss

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra


class YOLOModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(YOLOModule, self).__init__()

        self.save_hyperparameters()
        self.model = YOLO(
            num_classes=cfg.model.num_classes,
            in_channels=cfg.model.in_channels,
            backbone_channels=cfg.model.backbone_channels,
        )

        self.loss_fn = YOLOLoss(
            num_classes=cfg.model.num_classes,
            reg_max=cfg.model.reg_max,
            use_focal=cfg.training.use_focal_loss,
        )

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)

        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.training.lr)


@hydra.main(config_path="configs", config_name="train")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.training.seed)

    model = YOLOModule(cfg)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model)
