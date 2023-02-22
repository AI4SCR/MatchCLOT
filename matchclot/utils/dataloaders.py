import torch
from torch.utils.data import Dataset, DataLoader


class ModalityMatchingDataset(Dataset):
    def __init__(self, df_modality1, df_modality2):
        super().__init__()

        self.df_modality1 = df_modality1.values
        self.df_modality2 = df_modality2.values

    def __len__(self):
        return self.df_modality1.shape[0]

    def __getitem__(self, index: int):
        x_modality_1 = self.df_modality1[index]
        x_modality_2 = self.df_modality2[index]
        return {"features_first": x_modality_1, "features_second": x_modality_2}


def get_dataloaders(
    mod1_train,
    mod2_train,
    sol_train,
    mod1_test,
    mod2_test,
    sol_test,
    NUM_WORKERS,
    BATCH_SIZE,
):
    mod2_train = mod2_train.iloc[sol_train.values.argmax(1)]
    mod2_test = mod2_test.iloc[sol_test.values.argmax(1)]

    dataset_train = ModalityMatchingDataset(mod1_train, mod2_train)
    data_train = DataLoader(
        dataset_train, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    dataset_test = ModalityMatchingDataset(mod1_test, mod2_test)
    data_test = DataLoader(
        dataset_test, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return data_train, data_test


class AugDataset(Dataset):
    """
    Modality matching dataset class for training with affine transformation data augmentations. Given 4 affine
    transformations for each modality, it applies a random transformation which is an interpolation of the 4
    transformations. Can be used to augment a distribution of cell profiles from a single batch by applying affine
    transformations that approximate the optimal transport map between the non-augmented batch and other batches.
    """

    def __init__(
        self,
        df_modality1,
        df_modality2,
        transf_matrices_mod1,
        transf_vectors_mod1,
        transf_matrices_mod2,
        transf_vectors_mod2,
    ):
        """
        Args:
            df_modality1: pandas dataframe of mod1 data
            df_modality2: pandas dataframe of mod2 data
            transf_matrices_mod1: list of 4 linear transformation matrices for augmenting mod1
            transf_vectors_mod1: list of 4 translation vectors for augmenting mod1
            transf_matrices_mod2: list of 4 linear transformation matrices for augmenting mod2
            transf_vectors_mod2: list of 4 translation vectors for augmenting mod2
        """
        super().__init__()

        self.df_modality1 = df_modality1.values
        self.df_modality2 = df_modality2.values

        # save the affine transformations
        self.transf_matrices_mod1 = transf_matrices_mod1
        self.transf_vectors_mod1 = transf_vectors_mod1
        self.transf_matrices_mod2 = transf_matrices_mod2
        self.transf_vectors_mod2 = transf_vectors_mod2

        # define the tetrahedron
        n_dim = 3
        r1 = [1, 1, 1]
        r2 = [1, -1, -1]
        r3 = [-1, 1, -1]
        r4 = [-1, -1, 1]
        rfirst = [r1, r2, r3]
        # transformation matrix to get the barycentric coordinates
        T = torch.Tensor(
            [
                [rfirst[col][row] - r4[row] for col in range(n_dim)]
                for row in range(n_dim)
            ]
        )
        Tinv = torch.linalg.inv(T)
        # class attributes used by self._sample_mixing_coeff()
        self.r4 = torch.Tensor(r4)
        self.Tinv = Tinv

    def _sample_mixing_coeff(self):
        """
        Returns 4 mixing coefficients, which are the barycentric coordinates of a random point sampled through
        rejection sampling from a uniform distribution over the volume of a regular 3D tetrahedron. Coordinate
        conversion formulas from https://en.wikipedia.org/wiki/Barycentric_coordinate_system
        #Barycentric_coordinates_on_tetrahedra
        """
        # this loop has probability of exiting 1/3 (= tetrahedron volume / cube volume) at each iteration
        # therefore, the expected running time is 3 iterations
        while True:
            # sample a point from the uniform distribution on [-1, 1]^3
            pt = 2 * torch.rand(3) - 1
            # reject and resample the point if it's outside the tetrahedron
            if not (
                pt[0] < pt[1] + pt[2] - 1
                or pt[0] > -pt[1] + pt[2] + 1
                or pt[0] > pt[1] - pt[2] + 1
                or pt[0] < -pt[1] - pt[2] - 1
            ):
                break

        # transform from cartesian to barycentric coordinates
        c1, c2, c3 = (self.Tinv @ (pt - self.r4)).tolist()
        c4 = 1 - c1 - c2 - c3

        return c1, c2, c3, c4

    def __len__(self):
        return self.df_modality1.shape[0]

    def __getitem__(self, index: int):
        x_mod2 = self.df_modality1[index]
        x_mod1 = self.df_modality2[index]

        c1, c2, c3, c4 = self._sample_mixing_coeff()
        # sample a random augmentation strength uniformly between 0 and 1
        aug_strength = torch.rand(size=(1,))
        # use the same coefficients for the two modalities so the matching still makes sense
        x_mod1 = (
            aug_strength
            * (
                c1
                * (self.transf_matrices_mod1[0] @ x_mod1 + self.transf_vectors_mod1[0])
                + c2
                * (self.transf_matrices_mod1[1] @ x_mod1 + self.transf_vectors_mod1[1])
                + c3
                * (self.transf_matrices_mod1[2] @ x_mod1 + self.transf_vectors_mod1[2])
                + c4
                * (self.transf_matrices_mod1[3] @ x_mod1 + self.transf_vectors_mod1[3])
            )
            + (1 - aug_strength) * x_mod1
        )
        x_mod2 = (
            aug_strength
            * (
                c1
                * (self.transf_matrices_mod2[0] @ x_mod2 + self.transf_vectors_mod2[0])
                + c2
                * (self.transf_matrices_mod2[1] @ x_mod2 + self.transf_vectors_mod2[1])
                + c3
                * (self.transf_matrices_mod2[2] @ x_mod2 + self.transf_vectors_mod2[2])
                + c4
                * (self.transf_matrices_mod2[3] @ x_mod2 + self.transf_vectors_mod2[3])
            )
            + (1 - aug_strength) * x_mod2
        )
        return {"features_first": x_mod2.squeeze(), "features_second": x_mod1.squeeze()}


class SingleAugDataset(Dataset):
    """
    Similar to AugDataset class, but only applies a single affine transformation to each modality.
    """

    def __init__(
        self,
        df_modality1,
        df_modality2,
        transf_matrix_mod1,
        transf_vector_mod1,
        transf_matrix_mod2,
        transf_vector_mod2,
    ):
        """
        Args:
            df_modality1: pandas dataframe of mod1 data
            df_modality2: pandas dataframe of mod2 data
            transf_matrices_mod1: linear transformation matrix for augmenting mod1
            transf_vectors_mod1: translation vector for augmenting mod1
            transf_matrices_mod2: linear transformation matrix for augmenting mod2
            transf_vectors_mod2: translation vector for augmenting mod2
        """
        super().__init__()

        self.df_modality1 = df_modality1.values
        self.df_modality2 = df_modality2.values

        # save the affine transformations
        self.transf_matrix_mod1 = transf_matrix_mod1
        self.transf_vector_mod1 = transf_vector_mod1
        self.transf_matrix_mod2 = transf_matrix_mod2
        self.transf_vector_mod2 = transf_vector_mod2

    def __len__(self):
        return self.df_modality1.shape[0]

    def __getitem__(self, index: int):
        x_mod2 = self.df_modality1[index]
        x_mod1 = self.df_modality2[index]

        # sample a random augmentation strength uniformly between 0 and 1
        aug_strength = torch.rand(size=(1,))
        # use the same coefficients for the two modalities so the matching still makes sense
        x_mod1 = (
            aug_strength * (self.transf_matrix_mod1 @ x_mod1 + self.transf_vector_mod1)
            + (1 - aug_strength) * x_mod1
        )
        x_mod2 = (
            aug_strength * (self.transf_matrix_mod2 @ x_mod2 + self.transf_vector_mod2)
            + (1 - aug_strength) * x_mod2
        )

        return {"features_first": x_mod2.squeeze(), "features_second": x_mod1.squeeze()}
