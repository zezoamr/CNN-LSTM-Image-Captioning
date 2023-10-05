import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoLSTM
from numpy import average


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="flickr8k/images",
        captions_file="flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )
    
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = False
    save_every = 10
    print_every = 10

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0
    
    # initialize model, loss etc
    model = CNNtoLSTM(embed_size, hidden_size, vocab_size, num_layers, train_CNN).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()
    
    try:
        for epoch in range(num_epochs):
            
            if (epoch + 1) % print_every == 0:
                print_examples(model, device, dataset)
            
            if save_model and (epoch +1) % save_every == 0:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                save_checkpoint(checkpoint)
            
            loop = tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            
            for _, (imgs, captions) in loop:
                loop.set_description(f"Epoch {epoch}")
                
                imgs = imgs.to(device)
                captions = captions.to(device)
                outputs = model(imgs, captions[:-1])

                reshaped_outputs = outputs.reshape(-1, outputs.shape[2])
                reshaped_captions = captions.reshape(-1)

                loss = criterion(reshaped_outputs, reshaped_captions)

                writer.add_scalar("Training loss", loss.item(), global_step=step)
                step += 1

                optimizer.zero_grad()
                loss.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
    except KeyboardInterrupt:
        if save_model:
            checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
            save_checkpoint(checkpoint)    


if __name__ == "__main__":
    train()
