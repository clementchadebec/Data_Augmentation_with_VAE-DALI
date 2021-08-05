from dataclasses import dataclass


@dataclass
class ArgsConfig:
    path_to_train_loader: str = "trained_models/train_loader_MNIST"
    batch_size: int = 180
    max_epochs: int = 10000
    lr: float = 1e-3
    early_stopping_epochs: int = 50
    no_cuda: bool = False
    seed: int = 8
    model_name: str = "VAE"
    input_dim: int = 784
    latent_dim: int = 2
    n_lf: int = 3
    eps_lf: float = 0.001
    beta_zero: float = 0.3
    temperature: float = 0.8
    regularization: float = 0.01
    metric_fc: int = 400
    dynamic_binarization: bool = False
    verbose: bool = False
    device: str = "cpu"


@dataclass
class Model_config:
    input_size: int = None
    z1_size: int = None
    prior: str = None
    input_type: str = None
    use_training_data_init: bool = None
    pseudoinputs_mean: float = None
    pseudoinputs_std: float = None
    number_components: int = None
    cuda = None


@dataclass
class Train_config:
    epochs: int = None
    model_name: str = None
    early_stopping_epochs: int = None
    warmup: int = None
    dynamic_binarization: bool = None
    cuda: bool = None
