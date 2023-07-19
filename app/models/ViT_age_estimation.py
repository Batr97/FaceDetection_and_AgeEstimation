from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification


class OpenEyesClassificator():
    """
    OpenEyesClassificator is a class that uses a pretrained Vision Transformer (ViT) model for image classification.
    Args:
        model_path (str): The path to the pretrained ViT model.
    Attributes:
        model (ViTForImageClassification): The pretrained ViT model.
        feature_extractor (ViTFeatureExtractor): The feature extractor used to preprocess the input images.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the OpenEyesClassificator with a pretrained Vision Transformer model.
        Args:
            model_path (str): The path to the pretrained ViT model.
        """
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)

    def predict(self, inpIm: Image.Image) -> torch.Tensor:
        """
        Predicts the class probabilities for the input image.
        Args:
            inpIm (str): The file path of the input image.
        Returns:
            torch.Tensor: A tensor containing the class probabilities for the input image.
        """
        # img = Image.open(inpIm).convert('RGB')
        img_features = self.feature_extractor(inpIm, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**img_features).logits
        return logits
