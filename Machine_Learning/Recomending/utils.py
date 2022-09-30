import numpy as np
import scipy.sparse
import torch


def build_weight(*dimensions):
    weight = torch.nn.Parameter(torch.empty(*dimensions))
    torch.nn.init.xavier_normal_(weight)
    return weight


def build_bias(*dimensions):
    bias = torch.nn.Parameter(torch.zeros(*dimensions))
    return bias


def to_sparse_coo(tensor):
    try:
        return tensor.to_sparse_coo()
    except NotImplementedError:
        # Torch errors if coo tensor is cast to coo again.
        return tensor


def torch_sparse_to_scipy_coo(sparse_tensor):
    sparse_tensor = to_sparse_coo(sparse_tensor).coalesce().cpu()
    sparse_tensor = scipy.sparse.coo_matrix(
        (sparse_tensor.values().numpy(), sparse_tensor.indices().numpy()),
        shape=sparse_tensor.shape,
    )
    return sparse_tensor


def scipy_coo_to_torch_sparse(sparse_matrix, device=None):
    return torch.sparse_coo_tensor(
        indices=np.stack([sparse_matrix.row, sparse_matrix.col]),
        values=sparse_matrix.data,
        size=sparse_matrix.shape,
        device=device,
    )


def torch_sparse_slice(sparse_matrix, row_ids=None, col_ids=None, device=None):
    if torch.is_tensor(sparse_matrix):
        sparse_matrix = torch_sparse_to_scipy_coo(sparse_matrix)

    if row_ids is None:
        row_ids = slice(None)
    elif torch.is_tensor(row_ids):
        row_ids = row_ids.cpu().numpy()

    if col_ids is None:
        col_ids = slice(None)
    elif torch.is_tensor(col_ids):
        col_ids = col_ids.cpu().numpy()

    sparse_matrix = sparse_matrix.tocsr()[row_ids][:, col_ids].tocoo()
    torch_sparse_coo_tensor = scipy_coo_to_torch_sparse(sparse_matrix, device)
    return torch_sparse_coo_tensor


def split_dataset(dataset, fraction):
    len_dataset = len(dataset)
    right_size = int(fraction * len_dataset)
    left_size = len_dataset - right_size
    return torch.utils.data.random_split(dataset, [left_size, right_size])
