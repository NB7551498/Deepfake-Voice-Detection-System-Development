import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor

class SSLDeepfakeModel(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze_encoder=True):
        super(SSLDeepfakeModel, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Typically wav2vec2 outputs hidden states of size 768
        self.hidden_size = self.encoder.config.hidden_size
        
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is B x 1 x Samples -> remove channel dim for huggingface: B x Samples
        x = x.squeeze(1)
        
        # B x Time x Hidden
        outputs = self.encoder(x).last_hidden_state
        
        # Attention pooling
        att_weights = torch.softmax(self.attention(outputs), dim=1)
        # B x Hidden
        pooled_output = torch.sum(outputs * att_weights, dim=1)
        
        out = self.classifier(pooled_output)
        return out
