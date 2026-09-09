import torch
import torch.nn as nn
from src.models.cnn import MelSpectrogramCNN
from src.models.ssl_model import SSLDeepfakeModel

class DeepfakeEnsemble(nn.Module):
    def __init__(self, ssl_model_name="facebook/wav2vec2-base"):
        super(DeepfakeEnsemble, self).__init__()
        self.cnn = MelSpectrogramCNN()
        self.ssl = SSLDeepfakeModel(model_name=ssl_model_name)
        
        # Learnable weights for the ensemble, initialized evenly
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x):
        cnn_out = self.cnn(x)
        ssl_out = self.ssl(x)
        
        # Normalize weights to sum to 1
        normalized_weights = torch.softmax(self.weights, dim=0)
        
        ensemble_out = (normalized_weights[0] * cnn_out) + (normalized_weights[1] * ssl_out)
        return ensemble_out
