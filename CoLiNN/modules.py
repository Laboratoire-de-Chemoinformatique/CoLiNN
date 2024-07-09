from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from torch import nn
from torch.nn import (ModuleList, Linear, Module, LayerNorm)
from torch.nn import functional
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import GCNConv


class GraphEmbedding(Module):
    def __init__(self, batch_size, vector_dim: int, num_conv_layers: int, *args, **kwargs):
        super(GraphEmbedding, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.expansion = Linear(8, vector_dim)
        self.graph_convs = ModuleList([GCNConv(vector_dim, vector_dim) for _ in range(num_conv_layers)])
        self.norms = ModuleList(LayerNorm(vector_dim) for _ in range(num_conv_layers))

    def forward(self, pyg_graphs):
        atoms = pyg_graphs.x
        connections = pyg_graphs.edge_index
        atoms = torch.log(atoms + 1)
        atoms = self.expansion(atoms)

        for gcn_conv, norm in zip(self.graph_convs, self.norms):
            new_atoms = gcn_conv(atoms, connections)
            new_atoms = norm(new_atoms)
            new_atoms = functional.gelu(new_atoms)
            atoms = atoms + new_atoms

        graph_embedding = global_add_pool(atoms, pyg_graphs.batch)
        return graph_embedding


class CoLiNN(pl.LightningModule):
    def __init__(
            self,
            batch_size: int,
            vector_dim: int,
            num_conv_layers: int,
            num_reactions: int,
            num_gtm_nodes: int,
            lr: float,
            bbs_embed_path: str,
            bbs_pyg_path: str = None
    ):
        super(CoLiNN, self).__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.lr = lr
        self.num_reactions = num_reactions
        self.vector_dim = vector_dim
        self.num_gtm_nodes = num_gtm_nodes

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bbs_embed_path = Path(bbs_embed_path).resolve()
        if self.bbs_embed_path.exists():
            self.bbs_embeddings = torch.load(self.bbs_embed_path, map_location=device)
        else:
            bbs_pyg_path = Path(bbs_pyg_path).resolve(strict=True)
            self.building_blocks = torch.load(bbs_pyg_path, map_location=device)
            self.bbs_embeddings = torch.zeros(
                (len(self.building_blocks), self.vector_dim),
                dtype=torch.float32,
                device=device
            )

        # Layers initialisation
        self.graph_embeddings = GraphEmbedding(batch_size, vector_dim, num_conv_layers)
        self.reaction_embedding = nn.Embedding(num_reactions + 1, vector_dim,
                                               padding_idx=0)  # +1 since padding index is counted
        self.predictor = Linear(vector_dim, num_gtm_nodes)

    def forward(self, batch):
        graph_indices = batch[0].long()

        bbs_vectors = self.bbs_embeddings[graph_indices]

        reaction_ids = batch[1].long()
        reaction_vectors = self.reaction_embedding(reaction_ids)
        combined_vectors = torch.cat([reaction_vectors, bbs_vectors], dim=1)

        # Sum the vectors within each batch
        mol_vectors = combined_vectors.sum(dim=1)
        predictions = torch.softmax(self.predictor(mol_vectors), dim=1)

        return predictions

    def _get_loss(self, batch):
        true_y = batch[2].view(-1, self.num_gtm_nodes)  # responsibilities
        reaction_ids = batch[1].long()  # reaction ids
        reaction_vectors = self.reaction_embedding(reaction_ids)

        graph_indices = torch.flatten(batch[0]).long()  # bbs indices
        subgraphs = [self.building_blocks[i] for i in graph_indices.tolist()]
        # Batch the selected graphs
        batched_graphs = Batch.from_data_list(subgraphs)
        x = self.graph_embeddings(batched_graphs)

        # Mask out the -1 padding before reshaping the vectors
        mask = graph_indices != -1
        mask = mask.view(-1, 1).expand_as(x)
        x *= mask.float()  # Apply the mask
        # Save x (bb vectors before reshaping) for further use in forward
        if self.current_epoch == self.trainer.max_epochs - 1:
            self.bbs_embeddings[graph_indices] = x

        # Reshape the pooled graph vectors to match the batch size and number of graphs per batch
        bbs_vectors = x.view(-1, 3, x.size(1))

        combined_vectors = torch.cat([reaction_vectors, bbs_vectors], dim=1)
        # Sum the vectors within each batch
        mol_vectors = combined_vectors.sum(dim=1)
        predictions = self.predictor(mol_vectors)
        loss = F.kl_div(F.log_softmax(predictions, dim=1), true_y, reduction='batchmean')
        metrics = {"loss": loss}
        return metrics

    def on_train_end(self):
        # Code to execute at the end of fit
        torch.save(self.bbs_embeddings, self.bbs_embed_path)

    def training_step(self, batch, batch_idx):
        metrics = self._get_loss(batch)
        for name, value in metrics.items():
            self.log('train_' + name, value, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
            return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self._get_loss(batch)
        for name, value in metrics.items():
            self.log('val_' + name, value, on_epoch=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = AdaBelief(self.parameters(), lr=self.lr, eps=1e-16, betas=(0.9, 0.999),
                              weight_decouple=True, rectify=True, weight_decay=0.01,
                              print_change_log=False)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.8, min_lr=5e-5, verbose=True)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
