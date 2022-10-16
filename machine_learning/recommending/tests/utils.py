import numpy as np
import scipy


def get_config_base():
    config_base = dict(
        config_path="config_for_testing.yaml",
        logger=dict(name="WandbLogger", offline=True, anonymous=True, save_dir="local"),
        datamodule=dict(directory="../local/ml-100k", batch_size=100),
    )
    return config_base


def random_explicit_feedback(size, density=0.05, max_rating=1):
    ratings = np.arange(max_rating + 1)
    probs = [1 - density] + [density / max_rating] * max_rating
    return scipy.sparse.csr_matrix(np.random.choice(ratings, p=probs, size=size))
