from typing import Literal, Optional

import numpy as np

from ..core import BaseChemDataSet, BasePartitionCriteria

__all__ = [
    "BinaryRandomSplitCriterion",
    "TrainValidationTestSplitCriterion",
    "SequentialSplitCriterion",
    "StratifiedSplitCriterion",
]


class SplittingValueError(ValueError):
    pass


class SequentialSplitCriterion(BasePartitionCriteria):
    type: Literal["SequentialSplitCriterion"] = "SequentialSplitCriterion"
    labelA: str = "train"
    labelB: str = "test"
    labelB_size: float = 0.2

    def __call__(self, db: BaseChemDataSet) -> np.ndarray:
        return self.split_2_data_sequentially(len(db), self.labelA, self.labelB, self.labelB_size)

    @staticmethod
    def split_2_data_sequentially(
        len_db: int, labelA: str = "train", labelB: str = "test", labelB_size: float = 0.2
    ) -> np.ndarray:
        """
        Split the data into two labels in a sequential manner based on the fraction of the value range.
        The first values goes to labelB

        Args:
        len_db (int): The total number of data points.
        labelA (str): The label for the majority set. Defaults to "train".
        labelB (str): The label for the minority set. Defaults to "test".
        labelB_size (float): The fraction representing the size of the minority set.
                              Must be greater than 0 and less than 1. Defaults to 0.2.

        Returns:
        np.ndarray: An array of labels split sequentially based on the provided fraction.

        Raises:
        ValueError: If the fraction is not greater than 0 and less than 1.
        ValueError: If the provided fraction results in one empty set.
        """
        if not (0 < labelB_size < 1):
            raise ValueError("Fraction must be greater than 0 and less than 1.")

        data = list(range(len_db))
        divisor = int(1 / labelB_size)

        string_length = np.max([len(labelA), len(labelB)])
        if len_db == 1:
            return np.array([labelA], dtype=f"U{string_length}")
        split_labels = np.array([labelB if x % divisor == 0 else labelA for x in data], dtype=f"U{string_length}")
        # ret = list(sorted(set(E - np.geomspace(E, 1, N, dtype=np.int32) + 1)))

        # if len(set(split_labels)) != 2:
        #     raise SplittingValueError("The provided fraction results in one empty set. Adjust the fraction.")

        return split_labels


class BinaryRandomSplitCriterion(BasePartitionCriteria):
    type: Literal["BinaryRandomSplitCriterion"] = "BinaryRandomSplitCriterion"
    labelA: str = "train"
    labelB: str = "test"
    labelB_size: float = 0.2
    seed: Optional[int] = None
    weights_favors_B: Optional[BasePartitionCriteria] = None

    def __call__(self, db: BaseChemDataSet) -> np.ndarray:
        weights_favors_B_shaped = None
        if self.weights_favors_B is not None:
            weights_favors_B = self.weights_favors_B(db)
            weights_favors_B_shaped = weights_favors_B.reshape(len(db))
        return split_in2_datasets(
            len(db),
            labelA=self.labelA,
            labelB=self.labelB,
            labelB_size=self.labelB_size,
            seed=self.seed,
            weights_favors_B=weights_favors_B_shaped,
        )


class TrainValidationTestSplitCriterion(BasePartitionCriteria):
    type: Literal["TrainValidationTestSplitCriterion"] = "TrainValidationTestSplitCriterion"
    val_size: float = 0.0
    test_size: float = 0.0
    random_state: Optional[int] = None
    weights_favors_train: Optional[BasePartitionCriteria] = None
    label_train: str = "train_set"
    label_val: str = "validation_set"
    label_test: str = "test_set"

    def __call__(self, db: BaseChemDataSet) -> np.ndarray:
        weights_favors_train = None
        if self.weights_favors_train is not None:
            weights_favors_train = self.weights_favors_train(db)
            assert len(weights_favors_train.shape) == 2
            assert weights_favors_train.shape[1] == 1
            weights_favors_train = weights_favors_train.flatten()
        return split_random(
            len(db),
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state,
            weights_favors_train=weights_favors_train,
        )


class StratifiedSplitCriterion(BasePartitionCriteria):
    type: Literal["StratifiedSplitCriterion"] = "StratifiedSplitCriterion"
    stratify_criterion: BasePartitionCriteria
    labelA: str = "train"
    labelB: str = "test"
    labelB_size_per_cat: float = 0.2
    seed: Optional[int] = None

    def __call__(self, db: BaseChemDataSet) -> np.ndarray:
        stratify_values = self.stratify_criterion(db)
        return generate_stratified_labels(
            len(db),
            stratify_values=stratify_values.reshape(len(db)),
            labelA=self.labelA,
            labelB=self.labelB,
            labelB_size_per_cat=self.labelB_size_per_cat,
            seed=self.seed,
        )


def split_in2_datasets(
    len_db, labelA="train", labelB="test", labelB_size=0.2, seed=None, weights_favors_B=None
) -> np.ndarray:
    if not len_db > 0:
        raise ValueError("len_db must be larger than 1")
    string_length = np.max([len(labelA), len(labelB)])
    if len_db == 1:
        return np.array([labelA], dtype=f"U{string_length}")
    if seed is not None:
        np.random.seed(seed)

    if labelB_size >= 1:
        assert len_db >= labelB_size
        num_test = int(labelB_size)
    else:
        num_test = int(labelB_size * len_db)
        if num_test == 0:
            num_test = 1

    if weights_favors_B is not None:
        # Normalize weights to ensure they sum to 1
        weights = np.array(weights_favors_B).astype(float)
        weights /= weights.sum()
        test_indices = np.random.choice(len_db, size=num_test, replace=False, p=weights)
    else:
        test_indices = np.random.choice(len_db, size=num_test, replace=False)
    # Initialize all labels with labelA

    split_labels = np.array([labelA] * len_db, dtype=f"U{string_length}")
    # Set selected indices to labelB
    split_labels[test_indices] = labelB
    return split_labels


def split_random(
    len_db: int,
    val_size=0.0,
    test_size=0.0,
    random_state=None,
    weights_favors_train: Optional[np.ndarray] = None,
    label_train: str = "train_set",
    label_val: str = "validation_set",
    label_test: str = "test_set",
) -> np.ndarray:
    if test_size + val_size >= 1:
        raise ValueError("test_size and val_size together must be less than 1.")

    # simple case
    if not test_size > 0.0 or not val_size > 0.0:
        if not test_size > 0.0:
            labelB = label_val
            sizeB = val_size
        else:
            labelB = label_test
            sizeB = test_size
        split_labels = split_in2_datasets(
            len_db,
            labelA=label_train,
            labelB=labelB,
            labelB_size=sizeB,
            seed=random_state,
            weights_favors_B=weights_favors_train,
        )
        return split_labels

    # two step splitting
    split_labels = split_in2_datasets(
        len_db,
        labelA="__validation_test_sets__",
        labelB=label_train,
        labelB_size=1 - val_size - test_size,
        seed=random_state,
        weights_favors_B=weights_favors_train,
    )

    len_train_val = int(len_db * (val_size + test_size))
    test_size_relative = test_size / (val_size + test_size)
    split_labels_2 = split_in2_datasets(
        len_train_val,
        labelA=label_val,
        labelB=label_test,
        labelB_size=test_size_relative,
        seed=random_state,
    )

    mask = split_labels == "__validation_test_sets__"
    split_labels[mask] = split_labels_2
    return split_labels


def generate_stratified_labels(
    len_db, stratify_values: np.ndarray, labelA="train", labelB="test", labelB_size_per_cat=0.2, seed=None
) -> np.ndarray:
    """
    Generate stratified labels for splitting a dataset into 'train' and 'test'.

    Parameters:
    - len_db (int): Total number of elements in the dataset.
    - stratify_values (array): Categories for stratification.
    - labelB_size_per_cat (float or int): Proportion of the dataset to include in the test split.
    - random_state (int, optional): Seed for reproducibility.

    Returns:
    - labels (np.array): Array of labels ('train' or 'test') for each element.
    """
    if not len_db > 0:
        raise ValueError("len_db must be larger than 1")
    string_length = np.max([len(labelA), len(labelB)])
    if len_db == 1:
        return np.array([labelA], dtype=f"U{string_length}")
    if seed is not None:
        np.random.seed(seed)

    stratify_values = np.array(stratify_values)
    labels = np.array([""] * len(stratify_values), dtype=f"U{string_length}")
    for v in set(stratify_values):
        subset = np.where(stratify_values == v)[0]
        split = split_in2_datasets(
            len(subset), labelA=labelA, labelB=labelB, labelB_size=labelB_size_per_cat, seed=seed
        )
        for idx, split_i in zip(subset, split):
            labels[idx] = split_i

    return labels
