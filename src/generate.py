import torch
from torch.utils.data import DataLoader
from src.encoder_nano_risk import CausalEncoder  # Adjust import as necessary
from src.datamodule2 import LifeLightningDataModule  # Adjust import as necessary

class NextTokenGenerator:
    """
    Utility class for loading a trained CausalEncoder model, integrating with the dataloader, 
    and generating the next token.
    """
    def __init__(self, model_path: str, datamodule: LifeLightningDataModule, device: str = "cuda"):
        """
        Initializes the NextTokenGenerator with a trained model and dataloader.
        
        Args:
            model_path (str): Path to the trained model checkpoint.
            datamodule (LifeLightningDataModule): Lightning datamodule for handling data inputs.
            device (str): Device to load the model on ("cuda" or "cpu").
        """
        self.device = device
        self.datamodule = datamodule
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str) -> CausalEncoder:
        """
        Loads the model from a checkpoint.
        
        Args:
            model_path (str): Path to the checkpoint file.
            
        Returns:
            CausalEncoder: Loaded model.
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        model = CausalEncoder(**checkpoint['hyper_parameters'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        return model

    def generate_next_token(self, batch: dict) -> str:
        """
        Generates the next token for a given batch.
        
        Args:
            batch (dict): A batch of input data.
            
        Returns:
            str: The predicted next token.
        """
        with torch.no_grad():
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            output = self.model(batch)
            logits = self.model.decoder(output[:, -1, :])  # Only consider the last token output
            predicted_index = torch.argmax(logits, dim=-1).item()
            return self.datamodule.pipeline.vocab.idx_to_token[predicted_index]

    def predict_with_dataloader(self, dataloader: DataLoader):
        """
        Generates predictions for all batches in a dataloader.
        
        Args:
            dataloader (DataLoader): Dataloader providing batches of input data.
            
        Returns:
            List[str]: Predicted tokens for each batch.
        """
        predictions = []
        for batch in dataloader:
            token = self.generate_next_token(batch)
            predictions.append(token)
        return predictions


if __name__ == "__main__":
    # Path to the checkpoint
    model_checkpoint_path = "path/to/your/checkpoint.ckpt"

    # Instantiate the datamodule
    datamodule = LifeLightningDataModule(
        dir_path="path/to/your/data",
        sources=None,  # Provide sources as necessary
        background=None,  # Provide background as necessary
        cls_token=True,
        sep_token=True,
        segment=True,
        batch_size=32,
        num_workers=4,
        max_seq_len=128,
        cutoff=5,  # Adjust as per your dataset
    )

    datamodule.prepare_data()  # Prepare the data
    dataloader = datamodule.train_dataloader()  # Get the train dataloader

    # Initialize the generator
    generator = NextTokenGenerator(model_checkpoint_path, datamodule)

    # Generate predictions for a dataloader
    predictions = generator.predict_with_dataloader(dataloader)
    print(f"Predictions: {predictions}")
