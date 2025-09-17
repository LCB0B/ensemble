from typing import List

import polars as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter

from src.paths import FPATH


class SaveAllInfoWriter(BasePredictionWriter):
    def __init__(self, fname: str):
        """
        Custom writer to save predictions to a structured DataFrame.

        Args:
            fname (str): The name for the prediction file
            write_interval (str): "batch" or "epoch" to define when predictions are written.
        """
        super().__init__("batch_and_epoch")
        self.fname = fname
        self.accumulated_info = {}

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """
        Accumulates predictions from each batch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being used for prediction.
            prediction: The prediction result from the batch.
            batch_indices: The indices of the batch.
            batch: The actual batch data.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.
        """
        print(batch.keys())
        # print("Person id", batch['person_id'])
        if len(self.accumulated_info.keys()) == 0:
            for key in batch.keys():
                self.accumulated_info[key] = []
            self.accumulated_info["predictions"] = []

        # Process each key in the batch dictionary
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Convert tensor to a list
                self.accumulated_info[key].extend(value.cpu().tolist())
            else:
                self.accumulated_info[key].extend(value)

        self.accumulated_info["predictions"].extend(prediction.cpu().tolist())

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """
        Writes all accumulated predictions at the end of the epoch as a DataFrame.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being used for prediction.
            predictions: All accumulated predictions for the epoch.
            batch_indices: The indices of all batches for the epoch.
        """

        df = pl.DataFrame(self.accumulated_info)

        # Save the DataFrame to disk
        output_path = FPATH.DATA / f"{self.fname}.parquet"
        print("Writing predictions to local IO")
        df.write_parquet(output_path)
        print("Copying predictions to network drive")
        FPATH.copy_to_opposite_drive(output_path)

        # Optionally, clear accumulated info after saving
        self.accumulated_info.clear()


class SaveSelectiveInfo(BasePredictionWriter):
    def __init__(
        self,
        fname: str,
        keys_to_save: List[str] = ["person_id", "censor"],
        folder: str | None = None,
    ):
        """
        Custom writer to save predictions to a structured DataFrame.

        Args:
            fname (str): The name for the prediction file
            write_interval (str): "batch" or "epoch" to define when predictions are written.
        """
        super().__init__("batch_and_epoch")
        self.folder = folder
        self.fname = fname
        self.keys_to_save = keys_to_save
        self.accumulated_info = {}

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """
        Accumulates predictions from each batch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being used for prediction.
            prediction: The prediction result from the batch.
            batch_indices: The indices of the batch.
            batch: The actual batch data.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.
        """
        if len(self.accumulated_info.keys()) == 0:
            for key in self.keys_to_save:
                self.accumulated_info[key] = []
            self.accumulated_info["predictions"] = []

        # Process each key in the batch dictionary
        for key, value in batch.items():
            if key in self.keys_to_save:
                if isinstance(value, torch.Tensor):
                    # Convert tensor to a list
                    self.accumulated_info[key].extend(value.cpu().tolist())
                else:
                    self.accumulated_info[key].extend(value)

        self.accumulated_info["predictions"].extend(prediction.cpu().tolist())

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """
        Writes all accumulated predictions at the end of the epoch as a DataFrame.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being used for prediction.
            predictions: All accumulated predictions for the epoch.
            batch_indices: The indices of all batches for the epoch.
        """
        # DO THIS
        # self.accumulated_info['person_id'] = trainer.datamodule.predict_dataset.observations["person_id"]
        df = pl.DataFrame(self.accumulated_info)
        # Save the DataFrame to disk
        if self.folder:
            output_path = FPATH.DATA / self.folder / f"{self.fname}.parquet"
            output_path.parent.mkdir(exist_ok=True, parents=True)
        else:
            output_path = FPATH.DATA / f"{self.fname}.parquet"

        print("Writing predictions to local IO")
        print(output_path)
        df.write_parquet(output_path)
        print("Copying predictions to network drive")
        FPATH.alternative_copy_to_opposite_drive(output_path)

        # Optionally, clear accumulated info after saving
        self.accumulated_info.clear()


class SaveSimpleInfo(BasePredictionWriter):
    def __init__(self, fname: str):
        """
        Custom writer to save predictions to a structured DataFrame.

        Args:
            fname (str): The name for the prediction file
            write_interval (str): "batch" or "epoch" to define when predictions are written.
        """
        super().__init__("batch_and_epoch")
        self.fname = fname
        fname.parent.mkdir(parents=True, exist_ok=True)
        self.accumulated_info = {"predictions": [], "targets": [], "person_id": []}

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """
        Accumulates predictions from each batch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being used for prediction.
            prediction: The prediction result from the batch.
            batch_indices: The indices of the batch.
            batch: The actual batch data.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.
        """
        self.accumulated_info["person_id"].extend(batch["person_id"])
        self.accumulated_info["predictions"].extend(prediction.cpu())
        self.accumulated_info["targets"].extend(batch["target"].cpu())

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """
        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being used for prediction.
            predictions: All accumulated predictions for the epoch.
            batch_indices: The indices of all batches for the epoch.
        """
        # Save the DataFrame to disk
        output_path = self.fname
        torch.save(self.accumulated_info, output_path)
        # print("Copying predictions to network drive")
        # FPATH.alternative_copy_to_opposite_drive(output_path)

        # Optionally, clear accumulated info after saving
        self.accumulated_info.clear()


class SaveSimpleInfoRegression(BasePredictionWriter):
    def __init__(self, fname: str):
        """
        Custom writer to save predictions to a structured DataFrame.

        Args:
            fname (str): The name for the prediction file
            write_interval (str): "batch" or "epoch" to define when predictions are written.
        """
        super().__init__("batch_and_epoch")
        self.fname = fname
        fname.parent.mkdir(parents=True, exist_ok=True)
        self.accumulated_info = {"predictions": [], "targets": [], "person_id": []}

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """
        Accumulates predictions from each batch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being used for prediction.
            prediction: The prediction result from the batch.
            batch_indices: The indices of the batch.
            batch: The actual batch data.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.
        """
        self.accumulated_info["person_id"].extend(batch["person_id"])
        self.accumulated_info["predictions"].extend(prediction.cpu())
        self.accumulated_info["targets"].extend(
            batch["target_regression"].cpu()
        )  # CHANGED TO REGRESSION

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """
        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being used for prediction.
            predictions: All accumulated predictions for the epoch.
            batch_indices: The indices of all batches for the epoch.
        """
        # Save the DataFrame to disk
        output_path = self.fname
        torch.save(self.accumulated_info, output_path)
        # print("Copying predictions to network drive")
        # FPATH.alternative_copy_to_opposite_drive(output_path)

        # Optionally, clear accumulated info after saving
        self.accumulated_info.clear()
