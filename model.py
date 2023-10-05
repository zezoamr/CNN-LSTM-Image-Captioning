import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.train_cnn = train_cnn
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        for name, param in self.inception.named_parameters():
            if name =="fc.weight" or "fc.bias":
                param.requires_grad = True
            else:
                param.requires_grad = self.train_cnn
                
    def forward(self, images):
        inception_output = self.inception(images)
        if isinstance(inception_output, torch.Tensor):
            x = inception_output
        else:
            # If inception_output is an InceptionOutputs object, extract the logits tensor
            x = inception_output.logits
            
        return self.dropout(self.relu(x))
    
    
class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        features = features.unsqueeze(0)
        embeddings = torch.cat((features, embeddings))
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
class CNNtoLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, train_cnn=False):
        super(CNNtoLSTM, self).__init__()
        self.encoder = EncoderCNN(embed_size=embed_size, train_cnn=train_cnn)
        self.decoder = DecoderLSTM(embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size, num_layers=num_layers)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocab, max_length=50):
        result_caption = []
        with torch.no_grad():
            x = self.encoder(image)
            x = x.unsqueeze(0)
            states = None
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                outputs = self.decoder.linear(hiddens.squeeze(0))
                predicted = outputs.argmax(1)
                result_caption.append(predicted.item())
                
                x = self.decoder.embed(predicted)
                x = x.unsqueeze(0)
                if vocab.itos[predicted.item()] == '<EOS>':
                    break
                
        return [vocab.itos[idx] for idx in result_caption]