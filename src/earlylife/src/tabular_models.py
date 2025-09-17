from collections import Counter
from typing import List

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
from catboost import CatBoostClassifier
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.classification import AUROC

from src.earlylife.src.loggers import RetryTensorBoardLogger
from src.earlylife.src.paths import FPATH


def create_sparse_pipeline(
    numerical_cols: List[str],
    onehot_cols: List[str],
    count_cols: List[str],
    min_df: float,
    max_features: int,
) -> Pipeline:
    """
    Creates a sklearn pipeline that processes different column types according to their
    specification.

    Returns:
        Pipeline: A configured sklearn Pipeline.
    """
    # Transformer for numerical features with median imputation and scaling
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Transformer for one-hot encoding
    onehot_transformer = Pipeline(
        steps=[("onehotencoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Transformer for count vectorization
    count_transformers = [
        (
            f"count_{col}",
            Pipeline(
                steps=[
                    (
                        "countvectorizer",
                        CountVectorizer(
                            lowercase=False,
                            token_pattern=r"(?u)\b\w+\b",
                            min_df=min_df,
                            max_features=max_features,
                        ),
                    ),
                    ("scaler", MaxAbsScaler()),
                ]
            ),
            col,
        )
        for col in count_cols
    ]

    # Combine transformers for Logistic Regression
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("onehot", onehot_transformer, onehot_cols),
        ]
        + count_transformers,
        sparse_threshold=1,
    )

    # Create a pipeline that applies the preprocessor and then fits the classifier
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    return pipeline


class SparseDataset(Dataset):
    """Custom dataset that keeps data in sparse format and converts to dense on-the-fly."""

    def __init__(self, data: csr_matrix, labels: np.ndarray = None):
        """
        Args:
            data (csr_matrix): The input feature data in sparse format.
            labels (np.ndarray): The target labels (optional).
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Convert only the row for this index to dense format
        dense_row = torch.tensor(self.data[idx].toarray(), dtype=torch.float32).squeeze(
            0
        )

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return dense_row, label
        else:
            return dense_row  # Only return features for inference


class SparseDataModule(LightningDataModule):
    """DataModule for loading sparse matrix data."""

    def __init__(
        self,
        train_data: csr_matrix,
        train_labels: np.ndarray,
        val_data: csr_matrix,
        val_labels: np.ndarray,
        batch_size: int = 32,
        num_workers=0,
    ):
        """
        Args:
            train_data (csr_matrix): The training feature data in sparse format.
            train_labels (np.ndarray): The training target labels.
            val_data (csr_matrix): The validation feature data in sparse format.
            val_labels (np.ndarray): The validation target labels.
            batch_size (int): Batch size for the DataLoader.
        """
        super().__init__()
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        self.train_dataset = SparseDataset(self.train_data, self.train_labels)
        self.val_dataset = SparseDataset(self.val_data, self.val_labels)

        # Compute class weights for balanced sampling
        label_counts = Counter(self.train_labels)
        class_weights = {label: 1.0 / count for label, count in label_counts.items()}
        sample_weights = [class_weights[label] for label in self.train_labels]
        self.sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(self.train_labels), replacement=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class LogisticRegression(LightningModule):
    """Logistic regression model with elastic net regularization and AUC monitoring."""

    def __init__(
        self,
        input_dim: int,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        lr: float = 1e-3,
    ):
        """
        Args:
            input_dim (int): Number of input features.
            alpha (float): Regularization strength (overall penalty term).
            l1_ratio (float): The mix between L1 and L2 penalties. 0 is pure L2, 1 is pure L1.
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.auc_metric = AUROC(task="binary")

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Calculate binary cross-entropy loss
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

        # Elastic Net Penalty
        l1_penalty = torch.norm(self.linear.weight, p=1)
        l2_penalty = torch.norm(self.linear.weight, p=2) ** 2
        elastic_net_penalty = self.alpha * (
            self.l1_ratio * l1_penalty + (1 - self.l1_ratio) * l2_penalty
        )

        # Total loss
        loss += elastic_net_penalty
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

        # Calculate AUC
        auc_score = self.auc_metric(y_hat, y.int())
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        self.log("val_auc", auc_score, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, fused=True)

    @torch.compiler.disable()
    def log(self, *args, **kwargs):
        return super().log(*args, **kwargs)


# Objective function for Optuna
def objective_LR(
    trial,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    experiment_name: str,
    max_epochs: int = 30,
    batch_size: int = 8_192,
    num_workers: int = 32,
) -> float:
    """
    Objective function for Optuna to optimize a logistic regression model.

    Args:
        trial: The Optuna trial object to suggest hyperparameters.
        train_data (np.ndarray): Training feature data.
        train_labels (np.ndarray): Training labels.
        val_data (np.ndarray): Validation feature data.
        val_labels (np.ndarray): Validation labels.
        experiment_name (str): Name for saving checkpoints and logs.
        max_epochs (int): Maximum number of epochs to train the model.

    Returns:
        float: Best model score achieved during training.
    """
    # Define the hyperparameters to tune
    alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # Initialize DataModule and Model with hyperparameters from trial
    data_module = SparseDataModule(
        train_data,
        train_labels,
        val_data,
        val_labels,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = LogisticRegression(
        input_dim=train_data.shape[1], alpha=alpha, l1_ratio=l1_ratio, lr=lr
    )

    model = torch.compile(model)
    # Initialize logging, early stopping, and checkpointing
    log_folder = FPATH.BASELINE_LOGS / "lr_logs"
    log_folder.mkdir(exist_ok=True, parents=True)
    checkpoint_folder = FPATH.CHECKPOINTS / "logistic_regression" / experiment_name
    checkpoint_folder.mkdir(exist_ok=True)
    logger = RetryTensorBoardLogger(log_folder, name=experiment_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        dirpath=checkpoint_folder,
        filename=f"trial_{trial.number}_best",
        save_top_k=1,
        mode="max",
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Save the path of the best model checkpoint for this trial
    trial.set_user_attr("best_model_path", checkpoint_callback.best_model_path)

    return checkpoint_callback.best_model_score.item()


def objective_CatBoost(
    trial: optuna.trial.Trial,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    experiment: str,
    column_types: dict[str, list[str]],
    top_tokens_count: int = 2_500,
) -> float:
    """
    Objective function for Optuna to optimize a CatBoostClassifier model.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object to suggest hyperparameters.
        train_data (np.ndarray): Training feature data.
        train_labels (np.ndarray): Training labels.
        val_data (np.ndarray): Validation feature data.
        val_labels (np.ndarray): Validation labels.
        experiment (str): Name of the experiment for checkpoint.

    Returns:
        float: ROC AUC score achieved on the validation set.
    """
    # Define the hyperparameters based on the specified distributions
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
    random_strength = trial.suggest_int("random_strength", 1, 20)
    one_hot_max_size = trial.suggest_int("one_hot_max_size", 16, 256)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    leaf_estimation_iterations = trial.suggest_int("leaf_estimation_iterations", 1, 20)
    # colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.1, 0.5) # Not supported for gpu
    max_ctr_complexity = trial.suggest_int("max_ctr_complexity", 1, 2)
    depth = trial.suggest_int("depth", 4, 10)
    border_count = trial.suggest_int("border_count", 32, 128)

    text_processing = {
        "tokenizers": [
            {"tokenizer_id": "Space", "separator_type": "ByDelimiter", "delimiter": " "}
        ],
        "dictionaries": [
            {
                "dictionary_id": "Unigram",
                "gram_order": "1",
            }
        ],
        "feature_processing": {
            "default": [
                {
                    "dictionaries_names": ["Unigram"],
                    "feature_calcers": [f"BoW:top_tokens_count={top_tokens_count}"],
                    "tokenizers_names": ["Space"],
                }
            ]
        },
    }

    # Initialize the CatBoostClassifier with hyperparameters from trial
    classifier = CatBoostClassifier(
        cat_features=column_types[
            "nominal"
        ],  # Exclude ordinal, as tree-based models can handle ordinal as numeric features due to in-built non-linearity
        text_features=column_types["count"],
        auto_class_weights="Balanced",
        text_processing=text_processing,
        task_type="GPU",
        devices="0",
        learning_rate=learning_rate,
        max_ctr_complexity=max_ctr_complexity,
        random_strength=random_strength,
        one_hot_max_size=one_hot_max_size,
        l2_leaf_reg=l2_leaf_reg,
        bagging_temperature=bagging_temperature,
        leaf_estimation_iterations=leaf_estimation_iterations,
        # colsample_bylevel=colsample_bylevel, # Not supported for gpu
        border_count=border_count,
        depth=depth,
        # iterations=10, # For prototyping
        boosting_type="Plain",
        verbose=True,
        allow_writing_files=False,
        random_seed=73,
    )

    # Train the model
    classifier.fit(train_data, train_labels)

    # Predict probabilities for validation set and calculate ROC AUC
    val_probs = classifier.predict_proba(val_data)[:, 1]
    val_auc = roc_auc_score(val_labels, val_probs)

    experiment_checkpoint_folder = FPATH.CHECKPOINTS_CATBOOST / experiment
    experiment_checkpoint_folder.mkdir(exist_ok=True, parents=True)
    model_path = (
        experiment_checkpoint_folder / f"catboost_model_trial_{trial.number}.cbm"
    )

    # Saving model
    # if first
    print(trial.study.trials)
    if len(trial.study.trials) == 1:
        classifier.save_model(model_path)
        trial.set_user_attr("best_model_path", str(model_path))

    # Or best
    elif val_auc > trial.study.best_value:
        classifier.save_model(model_path)
        trial.set_user_attr("best_model_path", str(model_path))

    return val_auc
