from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.seed = 666

    config.test_only = False
    config.use_training_model = False
    config.train_state_path = "./cache/latest.trainstate"

    config.model_features = 128
    config.num_heads = 2
    config.num_encoders = 8
    config.num_register_tokens = 16

    config.cnn_dropout_rate = 0.2
    config.encoder_dropout_rate = 0.2
    config.pre_mlp_dropout_rate = 0.2

    config.train_batch_size = 40
    config.epoch_count = 200

    config.test_batch_size = 100

    config.init_learning_rate = 0.0
    config.peek_learning_rate = 0.001
    config.end_learning_rate = 0.0001
    config.warmup_steps = -1  # prior than warmup_steps_percent, default = total_steps // 10
    config.warmup_steps_percent = -1.0  # default use warmup_steps

    config.state_save_per_epoch = 10
    config.model_save_dir = "./cache"
    config.model_suffix = "model"

    config.use_graphic = True

    config.max_noise = 0.02
    config.salt_noise_prob = 0.01
    config.flip_prob = 0.5
    config.mixup_weight = 0.2
    config.transmix_weight = 0.2
    config.max_crop_width = 4
    config.max_crop_height = 4
    config.max_scale_size = 38

    config.eval_per_epoch = 1

    config.enable_optimization = True

    return config
