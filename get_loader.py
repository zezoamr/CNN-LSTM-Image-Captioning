import os  
import pandas as pd 
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  
from torch.utils.data import DataLoader, Dataset
from PIL import Image  
import torchvision.transforms as transforms

# https://www.kaggle.com/datasets/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb/
# flickr dataset link

class FlickrDataset(Dataset):
    def __init__(self, root_folder, captions_file, transform=None, freq_threshold=5):
        self.root_folder = root_folder
        self.transform = transform
        self.df = pd.read_csv(captions_file)
        
        self.captions = self.df['caption']
        self.images = self.df['image']
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())
    
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.root_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform != None: 
            img = self.transform(img)
            
        caption = self.captions[index]    
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)
    
    def __len__(self):
        return len(self.df)


# Download with: python -m spacy download en
spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[t] if t in self.stoi else self.stoi["<UNK>"] for t in tokenized_text] 
    
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

class MyCollate:
    def __init__(self, pad_idx) -> None:
        self.pad_idx = pad_idx #idx in vocab
        
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        
        return imgs, targets

def get_loader(root_folder, captions_file, transform=None, batchsize=32, num_workers=8, shuffle=True, pinMemory=True):
        
    dataset = FlickrDataset(root_folder, captions_file, transform)
    
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pinMemory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    
    return loader, dataset


if __name__ == "__main__":
    #print(len(pd.read_csv("flickr8k/captions.txt")))
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    loader, dataset = get_loader(
        "flickr8k/images/", "flickr8k/captions.txt", transform=transform
    )

    for idx, (imgs, captions) in enumerate(loader):
        if idx == 4: break #just see 4 batches
        print(imgs.shape)
        print(captions.shape)