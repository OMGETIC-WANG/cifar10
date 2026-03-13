import kagglehub
import pickle
import numpy as np
import os
import jax.numpy as jnp


def LoadCIFAR10Kaggle(cache_path: str, kaggle_cache_path: str):
    download_path = os.path.join(cache_path, kaggle_cache_path)

    filepath = kagglehub.dataset_download(
        "pankrzysiu/cifar10-python",
        "cifar-10-python.tar.gz",
        output_dir=download_path,
    )
    os.system(f"tar -xf {filepath} -C {os.path.dirname(filepath)}")
    batch_dir = os.path.join(os.path.dirname(filepath), "cifar-10-batches-py")

    train_datas = []
    train_labels = []

    for i in range(1, 6):
        batch_filepath = os.path.join(batch_dir, f"data_batch_{i}")
        with open(batch_filepath, "rb") as f:
            dic = pickle.load(f, encoding="bytes")
        data: np.ndarray = dic[b"data"]
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        label = np.array(dic[b"labels"])
        train_datas.append(data)
        train_labels.append(label)

    test_filepath = os.path.join(batch_dir, "test_batch")
    with open(test_filepath, "rb") as f:
        dic = pickle.load(f, encoding="bytes")
    x_test: np.ndarray = dic[b"data"]
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(dic[b"labels"])

    x_train = np.concatenate(train_datas)
    y_train = np.concatenate(train_labels)

    np.savez(os.path.join(cache_path, "train.npz"), x=x_train, y=y_train)
    np.savez(os.path.join(cache_path, "test.npz"), x=x_test, y=y_test)

    os.system(f"rm -r {download_path}")

    return (x_train, y_train), (x_test, y_test)


def LoadCIFAR10(cache_path: str = "./cache/datasets/cifar10"):
    if os.path.exists(os.path.join(cache_path, "train.npz")) and os.path.exists(
        os.path.join(cache_path, "test.npz")
    ):
        with np.load(os.path.join(cache_path, "train.npz")) as data:
            x_train = data["x"]
            y_train = data["y"]
        with np.load(os.path.join(cache_path, "test.npz")) as data:
            x_test = data["x"]
            y_test = data["y"]
        return (jnp.array(x_train) / 255.0, jnp.array(y_train)), (
            jnp.array(x_test) / 255.0,
            jnp.array(y_test),
        )
    else:
        (x_train, y_train), (x_test, y_test) = LoadCIFAR10Kaggle(cache_path, "kaggle-download")
        return (jnp.array(x_train) / 255.0, jnp.array(y_train)), (
            jnp.array(x_test) / 255.0,
            jnp.array(y_test),
        )


LABEL_NAME = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}
