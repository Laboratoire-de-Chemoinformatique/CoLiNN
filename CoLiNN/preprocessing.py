import logging
import os.path as osp

import numpy as np
import torch
from GTMtools.files import parse_svm_vectors, count_svm_columns, count_svm_lines
from indigo import Indigo, IndigoObject
from mendeleev import element
from pytorch_lightning import LightningDataModule
from safetensors.torch import save_file, load_file
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm


def calc_atoms_info(accepted_atoms: tuple) -> dict:
    """
    Given a tuple of accepted atoms, return a dictionary with the atom symbol as the key and a tuple of
    the period, group, subshell, and number of electrons as the value.

    :param accepted_atoms: tuple of strings
    :type accepted_atoms: tuple
    :return: A dictionary with the atomic number as the key and the period, group, shell, and number of
    electrons as the value.
    """
    mendel_info = {}
    shell_to_num = {'s': 1, 'p': 2, 'd': 3, 'f': 4}
    for atom in accepted_atoms:
        mendel_atom = element(atom)
        period = mendel_atom.period
        group = mendel_atom.group_id
        shell, electrons = mendel_atom.ec.last_subshell()
        mendel_info[atom] = (period, group, shell_to_num[shell[1]], electrons)
    return mendel_info


# This function should be optimized so that we do not iterate through atoms
def is_atom_in_ring(molecule: IndigoObject, atom_index: int):
    """
    Determine if a specific atom in a given molecule is part of a ring.

    Parameters:
    - molecule (IndigoObject): Molecule to check.
    - atom_index (int): Index of the atom to check.

    Returns:
    - int: Returns 0 if the atom is in a ring, 1 otherwise.
    """
    ring_atoms = set()

    for ring in molecule.iterateRings(0, molecule.countAtoms()):
        for atom in ring.iterateAtoms():
            ring_atoms.add(atom.index())

    return 1 if atom_index in ring_atoms else 0


def atom_to_vector(atom: IndigoObject, mendel_info: dict, molecule: IndigoObject):
    """
    Given an atom, return a vector of length 8 with the following information:

    1. Atomic number
    2. Period
    3. Group
    4. Number of electrons + atom's charge
    5. Shell
    6. Total number of hydrogens
    7. Whether the atom is in a ring
    8. Number of neighbors

    :param atom: atom IndigoObject
    :param mendel_info: a dictionary of the form {'C': (3, 1, 1, 2), 'O': (3, 1, 1, 2), ...}
    :param molecule: molecule IndigoObject
    :type mendel_info: dict
    :return: The vector of the atom.
    """

    vector = np.zeros(8, dtype=np.int8)
    period, group, shell, electrons = mendel_info[atom.symbol()]
    vector[0] = atom.atomicNumber()
    vector[1] = period
    vector[2] = group
    vector[3] = electrons + atom.charge()
    vector[4] = shell
    vector[5] = atom.countHydrogens()
    vector[6] = is_atom_in_ring(molecule, atom.index())
    neighbors = [neighbor for neighbor in atom.iterateNeighbors()]
    vector[7] = len(neighbors)
    return vector


def graph_to_atoms_vectors(molecule: IndigoObject, max_atoms: int, mendel_info: dict):
    """
        Given a molecule, it returns a vector of shape (max_atoms, 11) where each row is an atom and each
        column is a feature.

        :param molecule: The molecule to be converted to a vector
        :type molecule: IndigoObject
        :param max_atoms: The maximum number of atoms in the molecule
        :type max_atoms: int
        :param mendel_info: a dictionary containing the information about the Mendel system
        :type mendel_info: dict
        :return: The atoms_vectors array
    """
    atoms_vectors = np.zeros((max_atoms, 8), dtype=np.int8)
    for atom in molecule.iterateAtoms():
        atoms_vectors[atom.index()] = atom_to_vector(atom, mendel_info, molecule)

    return atoms_vectors


def graph_to_coo_matrix(molecule):
    mol_adj, edge_attr = [], []
    for bond in molecule.iterateBonds():
        source = bond.source().index()
        dest = bond.destination().index()
        mol_adj.append([source, dest])
        edge_attr.append(bond.bondOrder())
    return torch.tensor(mol_adj, dtype=torch.long), torch.tensor(edge_attr, dtype=torch.long)


def preprocess_building_blocks(file):
    accepted_atoms = ('C', 'N', 'S', 'O', 'Se', 'F', 'Cl', 'Br', 'I', 'B', 'P', 'Si')
    mendel_info = calc_atoms_info(accepted_atoms)
    indigo = Indigo()
    with open(file) as bbs:
        for line in bbs:
            smiles = line.strip()
            molecule = indigo.loadMolecule(smiles)
            molecule.dearomatize()
            mol_adj, edge_attr = graph_to_coo_matrix(molecule)
            mol_atoms_x = graph_to_atoms_vectors(molecule, molecule.countAtoms(), mendel_info)
            mol_pyg_graph = Data(
                x=torch.tensor(mol_atoms_x, dtype=torch.int8),
                edge_index=mol_adj.t().contiguous(),
                edge_attr=edge_attr
            )
            mol_pyg_graph = ToUndirected()(mol_pyg_graph)
            assert mol_pyg_graph.is_undirected()
            yield mol_pyg_graph


def process_bbs(file_name, name, processed_dir):
    processed_bbs = []
    for data in tqdm(preprocess_building_blocks(file_name)):
        processed_bbs.append(data)
    torch.save(processed_bbs, osp.join(processed_dir, f"{name}_bbs.pt"))


def process(file_path, processed_file_path, mode):
    num_mols = count_svm_lines(file_path)

    # initialise empty output_matrices
    logging.debug("Initialization of empty output matrices")
    mol_bb_ids = torch.ones((num_mols, 3), dtype=torch.int32) * -1
    mol_reactions_ids = torch.zeros((num_mols, 3), dtype=torch.int16)

    if mode == "training":
        num_svm_columns = count_svm_columns(file_path)
        logging.debug("Counted number of svm columns")
        resp_vectors = torch.zeros((num_mols, num_svm_columns), dtype=torch.float32)
        logging.debug("Initialized responsibility vectors")
        for mol_num, (reaction_bbs, svm_line) in tqdm(enumerate(parse_svm_vectors(file_path)), total=num_mols):
            logging.debug("started parsing svm file")
            reaction_seq_id, bb_ids = reaction_bbs.split("_")
            bb_ids = [int(i) for i in bb_ids.split(";")]
            reaction_ids = [int(i) for i in reaction_seq_id.split(";")]

            for i, bb_id in enumerate(bb_ids):
                mol_bb_ids[mol_num][i] = bb_id

            for i, rxn_id in enumerate(reaction_ids):
                mol_reactions_ids[mol_num][i] = rxn_id

            for column_id, column_value in svm_line.items():
                resp_vectors[mol_num][column_id - 1] = float(column_value)

        processed_dataset = {"bbs_ids": mol_bb_ids,
                             "reactions_ids": mol_reactions_ids,
                             "responsibilities": resp_vectors}

    else:
        with open(file_path) as file:
            for mol_num, line in tqdm(enumerate(file), total=num_mols):
                sline = line.strip()
                reaction_seq_id, bb_ids = sline.split("_")
                bb_ids = [int(i) for i in bb_ids.split(";")]
                reaction_ids = [int(i) for i in reaction_seq_id.split(";")]

                for i, bb_id in enumerate(bb_ids):
                    mol_bb_ids[mol_num][i] = bb_id

                for i, rxn_id in enumerate(reaction_ids):
                    mol_reactions_ids[mol_num][i] = rxn_id

            processed_dataset = {"bbs_ids": mol_bb_ids,
                                 "reactions_ids": mol_reactions_ids}

    logging.debug(f"How dataset looks like: {processed_dataset}")
    save_file(processed_dataset, processed_file_path)
    return processed_dataset


class CoLiNNDataset(Dataset):
    def __init__(self, data_root, name, mode):  # split,
        super(CoLiNNDataset, self).__init__()
        self.mode = mode
        self.name = name
        self.data_root = osp.abspath(data_root)
        self.processed_dir = osp.join(self.data_root, "processed")
        self.raw_dir = osp.join(self.data_root, "raw")
        self.processed_file_path = osp.join(self.processed_dir, f"{self.name}.safetensors")  # _{self.split}
        self.bb_file = osp.join(self.raw_dir, f"{self.name}_bbs.svm")
        self.raw_file = osp.join(self.raw_dir, f"{self.name}.svm")

        # Process combinatorial library
        if osp.exists(self.processed_file_path):
            logging.debug("Loading processed file.safetensors")
            self.data = load_file(self.processed_file_path)
        else:
            logging.debug("Started processing raw file")
            self.data = process(self.raw_file, self.processed_file_path, self.mode)

    def __getitem__(self, idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.mode == "training":
            bbs_ids, reactions_ids, responsibilities = self.data["bbs_ids"][idx], self.data["reactions_ids"][idx], \
                self.data["responsibilities"][idx]
            return bbs_ids.to(device), reactions_ids.to(device), responsibilities.to(device)
        else:
            bbs_ids, reactions_ids = self.data["bbs_ids"][idx], self.data["reactions_ids"][idx]
            return bbs_ids.to(device), reactions_ids.to(device)

    def __len__(self):
        return len(self.data["bbs_ids"])


class CoLiNNData(LightningDataset, LightningDataModule):

    def __init__(
            self,
            root: str,
            name: str,
            batch_size: int,
            mode: str,  # = "training"
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last: bool = True,
    ):

        train_dataset, val_dataset, pred_dataset = None, None, None
        self.batch_size = batch_size
        self.name = name
        self.root = root
        self.raw_dir = osp.join(self.root, "raw")
        self.raw_file = osp.join(self.raw_dir, f"{self.name}.svm")
        input_file = self.raw_file
        self.mode = mode

        if osp.exists(input_file):
            logging.debug("Creation of CombiLibDataset using the input svm")
            full_dataset = CoLiNNDataset(self.root, self.name, self.mode)
        else:
            raise ValueError(f"Input file does not exist at {input_file}")

        if mode == "training":
            logging.debug("Splitting on train and val datasets")
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size

            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                                      torch.Generator().manual_seed(42))
            print(f'train_size: {len(train_dataset)}')
            print(f'val_size: {len(val_dataset)}')

        else:
            pred_dataset = full_dataset

        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            pred_dataset=pred_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
