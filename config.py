from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.seed = 666

    config.test_only = False
    config.use_training_model = False
    config.train_state_path = "./cache/latest.trainstate"

    config.model_features = 64
    config.num_heads = 2
    config.num_encoders = 8

    config.train_batch_size = 40
    config.epoch_count = 200

    config.test_batch_size = 100

    config.learning_rate = 0.0001

    config.state_save_per_epoch = 10
    config.model_save_dir = "./cache"
    config.model_suffix = "model"

    config.use_graphic = True

    config.max_train_noise = 0.02
    config.flip_prob = 0.5

    config.mixup_weight = 0.0001
    config.mixup_epoch_begin = 10
    config.mixup_epoch_end = 60

    config.eval_per_epoch = 1

    config.enable_optimization = True

    return config
