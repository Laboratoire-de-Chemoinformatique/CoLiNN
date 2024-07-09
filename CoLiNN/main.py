import logging
import os.path as osp

import click
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, ModelSummary

from CoLiNN.modules import CoLiNN
from CoLiNN.preprocessing import CoLiNNData, process_bbs


def main_pipeline(config, mode):
    logging.basicConfig(filename=config["Train"]["log_file"], filemode="w", level=config["Train"]["log_level"],
        format='%(asctime)s:%(levelname)s:%(message)s')
    # Process building blocks
    name = config["Data"]["name"]
    bb_file = osp.join(f"{config['Data'][mode]['root_path']}/raw/", f"{name}_bbs.svm")
    processed_bb_path = config["Model"]["bbs_pyg_path"]
    if not osp.exists(processed_bb_path) and bb_file:
        logging.debug("Started preprocessing building blocks")
        process_bbs(bb_file, name, f"{config['Data'][mode]['root_path']}/processed/")

    data = CoLiNNData(root=config["Data"][mode]["root_path"], batch_size=config['Model']['batch_size'],
        name=config["Data"]["name"], mode=mode)

    if 'bbs_pyg_path' not in config['Model']:
        config['Model']['bbs_pyg_path'] = None

    if 'bbs_embed_path' not in config['Model']:
        config['Model']['bbs_embed_path'] = None

    seed_everything(42, workers=True)

    model = CoLiNN(batch_size=config['Model']['batch_size'], vector_dim=config['Model']['vector_dim'],
        num_reactions=config['Model']['num_reactions'], num_conv_layers=config['Model']['num_conv_layers'],
        num_gtm_nodes=config['Model']['num_gtm_nodes'], bbs_pyg_path=config['Model']['bbs_pyg_path'],
        bbs_embed_path=config['Model']['bbs_embed_path'], lr=config['Train']['lr'], )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint = ModelCheckpoint(dirpath=config["Train"]["weights_path"], filename=config["Train"]["weights_name"],
                                 monitor="val_loss", mode="min")
    swa = StochasticWeightAveraging(swa_lrs=config["Train"]["lr"], swa_epoch_start=0.95)

    trainer = Trainer(accelerator='gpu', devices=[0], max_epochs=config["Train"]["max_epochs"],
                      callbacks=[lr_monitor, checkpoint, swa, ModelSummary(2)], precision=16, gradient_clip_val=1.0,
                      log_every_n_steps=50)

    if mode == "training":
        trainer.fit(model, data)
    else:
        model = CoLiNN.load_from_checkpoint(
            checkpoint_path=f"{config['Train']['weights_path']}/{config['Train']['weights_name']}.ckpt",
            batch_size=config['Model']['batch_size'], vector_dim=config['Model']['vector_dim'],
            num_reactions=config['Model']['num_reactions'], num_conv_layers=config['Model']['num_conv_layers'],
            num_gtm_nodes=config['Model']['num_gtm_nodes'], bbs_pyg_path=config['Model']['bbs_pyg_path'],
            bbs_embed_path=config['Model']['bbs_embed_path'], lr=config['Train']['lr'], )
        model.to(device)
        predictions = trainer.predict(model, datamodule=data)
        all_predictions = torch.cat(predictions, dim=0)
        torch.save(all_predictions, osp.join(config["Data"][mode]["root_path"], 'processed/predictions.pt'))


@click.command()
@click.option("--config", "config_path", required=True, help="Path to the config YAML file.",
    type=click.Path(exists=True), )
@click.option("--mode", default="training", type=click.Choice(["training", "prediction"]),
    help="Mode of operation (training or prediction).", )
def main(config_path, mode):
    config_path = osp.abspath(config_path)
    if osp.exists(config_path) and osp.isfile(config_path):
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        if config is not None:
            main_pipeline(config, mode)
    else:
        raise ValueError("The config file does not exist.")


if __name__ == '__main__':
    main()
