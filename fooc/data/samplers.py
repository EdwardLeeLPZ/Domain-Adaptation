from copy import deepcopy
import itertools
import random
from typing import Dict, List, Optional

from torch.utils.data.sampler import Sampler

from detectron2.utils import comm

class EquallyDatasetsTrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    This sample balances the provided datasets based on the indices mapping.
    """

    def __init__(
        self, dataset_dicts: List[Dict], dataset_name_to_indices: Dict[str, List[int]],
        dataset_selection_type: str = 'simple', shuffle: Optional[bool] = True, seed: Optional[int] = None,
    ):
        """
        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            dataset_name_to_indices (dict[str, list]): mapping from dataset name to list of corresponding indices
            dataset_selection_type (str): strategy how the next dataset is chosen.
                'simple': the datasets just alternates -> can lead to case Dataset 1 always on GPU 1, DS 2 on GPU 2, aso
                'per_gpu': the datasets alternates per gpu
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        # sanity check
        assert len(dataset_dicts) > 0, dataset_dicts
        assert dataset_selection_type in {'simple', 'per_gpu', 'per_gpu_6_2'}, dataset_selection_type

        indices = set().union(*dataset_name_to_indices.values())
        assert (
            indices == set(range(len(dataset_dicts)))), (
            f"There is a missmatch between dataset_dict indices and the provided "
            f"indices: {indices} != {set(range(len(dataset_dicts)))}"
        )

        self._dataset_selection_type = dataset_selection_type
        self._dataset_dicts = dataset_dicts
        self._dataset_name_to_indices = dataset_name_to_indices
        self._dataset_names = list(dataset_name_to_indices.keys())
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        if self._dataset_selection_type == 'simple':
            infinite_indices = self._alternating_infinite_indices()
        elif self._dataset_selection_type == 'per_gpu':
            infinite_indices = self._gpu_alternating_infinite_indices()
        elif self._dataset_selection_type == 'per_gpu_6_2':
            infinite_indices = self._gpu_alternating_infinite_indices_6_2()

        yield from itertools.islice(infinite_indices, start, None, self._world_size)

    def _alternating_infinite_indices(self):
        """Iterates over the datasets not considering gpus.
        Note: this can result that one gpu does only see one of multiple datasets
        """
        random.seed(self._seed)

        dataset_name_to_indices = deepcopy(self._dataset_name_to_indices)
        dataset_name_to_iter = {name: iter([]) for name in self._dataset_name_to_indices.keys()}
        while True:
            for dataset_name in self._dataset_names:
                # if dataset indices are over -> reshuffle
                iterator = dataset_name_to_iter[dataset_name]
                try:
                    index = next(iterator)
                except StopIteration:
                    indices = dataset_name_to_indices[dataset_name]
                    if self._shuffle:
                        random.shuffle(indices)
                    dataset_name_to_iter[dataset_name] = iter(indices)
                    index = next(dataset_name_to_iter[dataset_name])

                yield index

    def _gpu_alternating_infinite_indices(self):
        """Iterates over the datasets for each gpu separately.
        """
        random.seed(self._seed)
        num_gpus = self._world_size

        dataset_name_to_indices = deepcopy(self._dataset_name_to_indices)
        dataset_name_to_iter = {name: iter([]) for name in self._dataset_name_to_indices.keys()}
        rank_to_dataset_index = {rank_idx: rank_idx % len(self._dataset_names) for rank_idx in range(num_gpus)}
        while True:
            for rank in range(num_gpus):
                # get next dataset name and update dataset index
                dataset_index = rank_to_dataset_index[rank]
                dataset_name = self._dataset_names[dataset_index]
                rank_to_dataset_index[rank] = (dataset_index + 1) % len(self._dataset_names)

                # if dataset indices are over -> reshuffle
                iterator = dataset_name_to_iter[dataset_name]
                try:
                    index = next(iterator)
                except StopIteration:
                    indices = dataset_name_to_indices[dataset_name]
                    if self._shuffle:
                        random.shuffle(indices)
                    dataset_name_to_iter[dataset_name] = iter(indices)
                    index = next(dataset_name_to_iter[dataset_name])

                yield index

    def _gpu_alternating_infinite_indices_6_2(self):
        """Iterates over the datasets for each gpu separately.
        Half of the gpus will only use the second dataset while the other gpus will alternate between the datasets.
        """
        random.seed(self._seed)
        num_gpus = self._world_size

        dataset_name_to_indices = deepcopy(self._dataset_name_to_indices)
        dataset_name_to_iter = {name: iter([]) for name in self._dataset_name_to_indices.keys()}
        rank_to_dataset_index = {rank_idx: rank_idx % len(self._dataset_names) for rank_idx in range(num_gpus)}
        while True:
            for rank in range(num_gpus):
                # get next dataset name and update dataset index
                if rank % 2 == 1:
                    dataset_index = rank_to_dataset_index[rank]
                else:
                    dataset_index = 1  # we assume that the source dataset is given as second dataset
                dataset_name = self._dataset_names[dataset_index]
                if rank % 2 == 1:
                    rank_to_dataset_index[rank] = (dataset_index + 1) % len(self._dataset_names)

                # if dataset indices are over -> reshuffle
                iterator = dataset_name_to_iter[dataset_name]
                try:
                    index = next(iterator)
                except StopIteration:
                    indices = dataset_name_to_indices[dataset_name]
                    if self._shuffle:
                        random.shuffle(indices)
                    dataset_name_to_iter[dataset_name] = iter(indices)
                    index = next(dataset_name_to_iter[dataset_name])

                yield index