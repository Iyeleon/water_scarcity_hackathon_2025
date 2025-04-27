import numpy as np
from sklearn.model_selection._split import _BaseKFold

class ContiguousGroupKFold(_BaseKFold):
    def __init__(self, n_splits=5):
        super().__init__(n_splits, random_state = None, shuffle = False)

    def split(self, X, y = None, groups = None):
        if groups is None:
            raise ValueError("The 'groups' parameter is required for ContiguousGroupKFold.")

        # get unique groups
        unique_groups = sorted(np.unique(groups))
        n_groups = len(unique_groups)

        # ensure n_splits is less than or equal to num_groups
        if self.n_splits > n_groups:
            raise ValueError(f"Cannot have number of splits = {self.n_splits} greater than the number of groups = {n_groups}.")

        fold_sizes = np.full(self.n_splits, n_groups // self.n_splits, dtype=int)
        fold_sizes[:n_groups % self.n_splits] += 1

        group_starts = np.cumsum(np.insert(fold_sizes, 0, 0))[:-1]

        for i in range(self.n_splits):
            test_start = group_starts[i]
            test_end = test_start + fold_sizes[i]
            test_groups = unique_groups[test_start:test_end]

            test_mask = np.isin(groups, test_groups)
            train_mask = ~test_mask

            yield np.where(train_mask)[0], np.where(test_mask)[0]

# class ContiguousTimeSeriesSplit(_BaseKFold):
#     def __init__(self, n_splits=5, min_train_size=0.5, test_size=1):
#         """
#         Custom time series splitter that performs contiguous splits based on ordered groups.

#         Parameters:
#         - n_splits (int): Total number of possible splits. Controls how far the training window can slide.
#         - min_train_size (float or int): Minimum number or fraction of groups to start training with.
#         - test_size (int): Number of groups to include in the validation/test set for each split.
#         """
#         super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
#         self.min_train_size = min_train_size
#         self.test_size = test_size

#     def split(self, X, y=None, groups=None):
#         if groups is None:
#             raise ValueError("The 'groups' parameter is required for ContiguousTimeSeriesSplit.")

#         unique_groups = np.array(sorted(np.unique(groups)))
#         n_groups = len(unique_groups)

#         if isinstance(self.min_train_size, float):
#             if not (0 < self.min_train_size < 1):
#                 raise ValueError("min_train_size must be a float between 0 and 1.")
#             min_train_groups = int(np.ceil(self.min_train_size * n_groups))
#         elif isinstance(self.min_train_size, int):
#             min_train_groups = self.min_train_size
#         else:
#             raise ValueError("min_train_size must be an int or float.")

#         if self.test_size > self.n_splits:
#             raise ValueError("test_size must be less than or equal to n_splits.")

#         max_start = n_groups - min_train_groups - self.test_size + 1
#         if self.n_splits > max_start:
#             raise ValueError(
#                 f"n_splits too large for dataset size with given min_train_size and test_size. "
#                 f"Maximum allowed splits: {max_start}"
#             )

#         for i in range(self.n_splits):
#             train_end = min_train_groups + i
#             test_start = train_end
#             test_end = test_start + self.test_size

#             if test_end > n_groups:
#                 break  # not enough data for test fold

#             train_groups = unique_groups[:train_end]
#             test_groups = unique_groups[test_start:test_end]

#             train_mask = np.isin(groups, train_groups)
#             test_mask = np.isin(groups, test_groups)

#             yield np.where(train_mask)[0], np.where(test_mask)[0]

class ContiguousTimeSeriesSplit(_BaseKFold):
    def __init__(self, n_splits=5, min_train_size=0.5):
        """
        Custom splitter for time series data using contiguous group-based splitting.
        For each split, validation is all subsequent groups after training.

        Parameters:
        - n_splits (int): Number of training splits to perform.
        - min_train_size (int or float): Minimum number or fraction of groups to start training with.
        """
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.min_train_size = min_train_size

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("The 'groups' parameter is required for ContiguousTimeSeriesSplit.")

        unique_groups = np.array(sorted(np.unique(groups)))
        n_groups = len(unique_groups)

        # Determine starting train size
        if isinstance(self.min_train_size, float):
            if not (0 < self.min_train_size < 1):
                raise ValueError("min_train_size must be a float between 0 and 1.")
            min_train_groups = int(np.ceil(self.min_train_size * n_groups))
        elif isinstance(self.min_train_size, int):
            min_train_groups = self.min_train_size
        else:
            raise ValueError("min_train_size must be an int or float.")

        max_possible_splits = n_groups - min_train_groups
        if self.n_splits > max_possible_splits:
            raise ValueError(
                f"Cannot create {self.n_splits} splits with min_train_size={min_train_groups} "
                f"and {n_groups} total groups. Max possible splits: {max_possible_splits}"
            )

        for i in range(self.n_splits):
            train_end = min_train_groups + i
            train_groups = unique_groups[:train_end]
            val_groups = unique_groups[train_end:]

            train_mask = np.isin(groups, train_groups)
            val_mask = np.isin(groups, val_groups)

            yield np.where(train_mask)[0], np.where(val_mask)[0]