import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        #self.init_weights()
    
    def hidden_initialize(self, batch_size):
        return (torch.zeros((1, batch_size, self.hidden_size), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
),torch.zeros((1, batch_size, self.hidden_size), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
))
    
    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        captions = captions[:,:-1]
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        self.hidden = self.hidden_initialize(batch_size) 
        output , self.hidden = self.lstm(embeddings, self.hidden)
        outputs = self.linear(output)
        return outputs
    
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        #inputs = inputs.unsqueeze(1)
        batch_size = inputs.shape[0]
        print("Batch_size  " , batch_size)
        hidden = self.hidden_initialize(batch_size)
        for i in range(max_len):                                      # maximum sampling length
            lstm_states, hidden = self.lstm(inputs, hidden) 
            outputs = self.linear(lstm_states.squeeze(1)) 
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.cpu().numpy()[0].item())
            if predicted == 1:### removing the end token
                break
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids
        