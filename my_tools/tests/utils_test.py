import pathlib
import shutil

import torch

import my_tools.utils as utils


def test_free_cuda():
    memory_before = torch.cuda.mem_get_info()
    utils.free_cuda()
    memory_after = torch.cuda.mem_get_info()
    assert memory_after >= memory_before


def test_reuse_shelved_objects():
    obj = utils.reuse_shelved_object_or_construct(
        hashable_attribute=b"test",
        object_constructor=lambda: "test",
        object_name="test",
        dir_path="local",
    )
    assert obj == "test"

    path = pathlib.Path("local")
    assert path.exists()

    reused_obj = utils.reuse_shelved_object_or_construct(
        hashable_attribute=b"test",
        object_constructor=lambda: "assert_this_is_unused",
        object_name="test",
        dir_path="local",
    )
    assert obj == reused_obj

    different_obj = utils.reuse_shelved_object_or_construct(
        hashable_attribute=b"test",
        object_constructor=lambda: "assert_neq_test",
        object_name="wrong_name_but_same_hash",
        dir_path="local",
    )
    assert obj != different_obj

    shutil.rmtree("local")


def test_sparse_dense_multiply():
    i = [[0, 1, 1], [2, 0, 2]]
    v = [3, 4, 5]
    s = torch.sparse_coo_tensor(i, v, (2, 3))

    s_squared = utils.sparse_dense_multiply(s, s.to_dense())
    assert s_squared.is_sparse
    assert (s_squared.to_dense() == s.to_dense() ** 2).all()
