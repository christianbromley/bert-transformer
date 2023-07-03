import argparse
from pathlib import Path
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM, AdamW
from tokenizers import ByteLevelBPETokenizer
import torch
from tqdm.auto import tqdm

# Create the parser
parser = argparse.ArgumentParser(description='Convert PDF to .txt.')

# Add arguments
parser.add_argument('tokenizer_dir', type=str, help='Directory of the tokenizer')
parser.add_argument('text_dir', type=str, help='Directory of the text files')
parser.add_argument('output_dir', type=str, help='Directory of your output model')


def parse_arguments():
    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    tokenizer_dir = args.tokenizer_dir
    output_dir = args.output_dir
    text_dir = args.text_dir

    return tokenizer_dir, output_dir, text_dir


def initialize_tokenizer(tokenizer_dir, text_dir):
    # initialize the tokenizer using the tokenizer we created
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_dir, max_len=512)

    # get the file paths
    paths = [str(x) for x in Path(text_dir).glob('**/*.txt')]

    return tokenizer, paths

def masked_language_model(tensor):
    rand = torch.rand(tensor.shape)
    mask_arr = (rand < 0.15) * (tensor > 2)
    for i in range(tensor.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        tensor[i,selection] = 4
    return tensor

def build_tensors(paths, tokenizer):
    # our three tensors are:
    ## input ids - token IDs with a % of tokens masked with the mask token ID which in our case is 4
    input_ids = []
    ## mask - this is a binary tensor of 1 and 0 indicating where the masks are
    mask = []
    ## labels - these are just the unmasked token IDs
    labels = []
    # initialise the output
    tokenized_text = {}
    # loop through paths
    for path in paths:
        # read file
        with open(path, 'r', encoding='utf-8') as fp:
            lines = fp.read().split('\n')
        # get the file name
        fname = path.split('/')[-1]
        # tokenize the text in these lines
        sample = tokenizer(lines, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        #print(sample.shape)
        # save to the output dictionary
        tokenized_text[fname] = sample
        # the sample object contains some of our tensors - extract these
        ## get the input IDs and append to labels
        labels.append(sample.input_ids)
        ## get the attention mask - the binary
        mask.append(sample.attention_mask)
        ## now apply the masked language model function on the input IDs to mask 15% of tokens
        mlm_on_input = masked_language_model(sample.input_ids.detach().clone())
        input_ids.append(mlm_on_input)
        #print(path)

    # construct the output
    encodings = {
        'input_ids': input_ids,
        'mask': mask,
        'labels': labels
    }

    print(encodings['input_ids'])
    return tokenized_text, encodings


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return self.encodings['input_ids'][0].shape[0]
    def __getitem__(self, i):
        return {
            'input_ids': self.encodings['input_ids'][0][i]
        }


def create_data_loader(encodings):
    dataset = Dataset(encodings)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    return data_loader


def train_model(config, loader):
    # init model
    model = RobertaForMaskedLM(config)
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # and move our model over to the selected device
    model.to(device)
    # activate training mode
    model.train()
    # initialize optimizer
    optim = AdamW(model.parameters(), lr=1e-4)
    # set number of epochs
    epochs = 2
    # loop through epochs
    for epoch in range(epochs):
        print(f'Epoch {str(epoch)}')
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            print(batch.keys())
            input_ids = batch['input_ids'][0].to(device)
            attention_mask = batch['mask'][0].to(device)
            labels = batch['labels'][0].to(device)
            # model
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            # extract the loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())


def main():
    # parse args
    tokenizer_dir, output_dir, text_dir = parse_arguments()

    # init tokenizer
    tokenizer, paths = initialize_tokenizer(tokenizer_dir, text_dir)

    # tokenize text
    tokenized_text, encodings = build_tensors(paths, tokenizer)

    print(encodings['input_ids'])

    # data loader creation
    loader = create_data_loader(encodings)

    # build RoBERTa config
    config = RobertaConfig(
        vocab_size=30_522,  # we align this to the tokenizer vocab_size
        max_position_embeddings=514,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1
    )

    train_model(config, loader)

    model.save_pretrained(f'{output_dir}/thesis_bert')


if __name__ == "__main__":
     main()