import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import utils

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from dataset_sentinel import SenDataset, TestDataset
from SenCLIP import SenCLIP


# Define data loading and preprocessing here
def get_args():
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)

    # Data params
    parser.add_argument('--root_data_dir', type=str, default='/path/to/Sentinel_LUCAS/')
    parser.add_argument('--emb_path', type=str, default='/path/to/Lucas_Frozen_Embeddings/lucas_clipemb_RN50.pt')
    parser.add_argument('--data_path_list', type=str, default='/path/to/sentinel_paths.npy')
    parser.add_argument('--version_fold', type=str, default='SenCLIP')
    parser.add_argument('--dataset', type=str, default='test')  #SenDataset

    # Trainer hyperparameters
    parser.add_argument('--BATCH_SIZE', type=int, default=32)
    parser.add_argument('--NUM_WORKERS', type=int, default=8)
    parser.add_argument('--NUM_EPOCHS', type=int, default=10)
    parser.add_argument('--ARCH', type=str, default='RN50')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--OPT', type=str, default='adamw')
    parser.add_argument('--pooling', type=str, default='avgpool')  # Options: avgpool, attpool_perdim, attpool_perimage
    parser.add_argument('--pool_out', type=str, default='sum')
    parser.add_argument('--trainable_layers', nargs='+', default=["visual"])

    # Model hyperparameters
    parser.add_argument('--LR', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    # Checkpoint and resume
    parser.add_argument('--saved_model', type=str, default='/path/to/checkpoint folder/')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--id', type=str, default=None)

    return parser.parse_args()


# Compute cross-entropy contrastive loss
def contrastive_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, labels)


# Main training loop
def main(args):
    class SenCLIPlearner(pl.LightningModule):
        def __init__(self, **kwargs):
            super().__init__()
            
            self.save_hyperparameters()

            # Model configuration
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / kwargs["temperature"]))
            self.learner = SenCLIP(
                embed_dim=1024 if kwargs["architecture"] == "RN50" else 512,
                architecture=kwargs["architecture"],
                trainable_layers = args.trainable_layers,
                pooling=kwargs["pooling"],
                device=kwargs["device"],
                pool_out=kwargs["pool_out"],
                queue_size=kwargs["queue_size"],
                queue_data=kwargs["queue_data"],
            )

        def training_step(self, batch, batch_idx):
            logits, labels = self.learner(batch)
            logit_scale = self.logit_scale.exp()
            logits_scaled = logits * logit_scale
            loss = contrastive_loss(logits_scaled, labels)
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            return loss

        def configure_optimizers(self):
            if self.hparams.optimiser == "sgd":
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.hparams.learning_rate,
                    momentum=0.9,
                    weight_decay=self.hparams.weight_decay,
                )
            elif self.hparams.optimiser == "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
                )
            else:
                optimizer = torch.optim.Adam(
                    self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
                )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
            return [optimizer], [scheduler]

        def on_before_zero_grad(self, *args, **kwargs):
            self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)

    # Instantiate the model
    model = SenCLIPlearner(
        batch_size=args.BATCH_SIZE,
        learning_rate=args.LR,
        architecture=args.ARCH,
        optimiser=args.OPT,
        queue_size=512,
        queue_data=None,
        pooling=args.pooling,
        pool_out=args.pool_out,
        temperature=0.07,
        device = args.device
    )

    # Logger and Callbacks
    wandb_logger = WandbLogger(name=f"{args.version_fold}_train", project="Granular", id=args.id if args.resume else None)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.saved_model}/{args.version_fold}/",
        filename=f"{args.version_fold}" + "-{epoch:02d}-{train_loss:.2f}",
        save_last=True,
        save_top_k=3,
        monitor="train_loss",
    )

    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=args.NUM_EPOCHS,
        devices=[args.device],
        accelerator="gpu",
        callbacks=[lr_monitor, checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model)


if __name__ == "__main__":
    args = get_args()

    # Dataset loading
    if args.dataset == "test":
        train_dataset = TestDataset()
    else:
        train_dataset = SenDataset(args.root_data_dir, args.emb_path, args.data_path_list)

    print("Sample input shapes:", train_dataset[0][0].shape, train_dataset[0][1].shape)
    main(args)
