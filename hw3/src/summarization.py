import argparse
import torch
from torchtext.data import BucketIterator

from model import EncoderDecoder
from utilities import DEVICE, load_word_field, load_dataset, make_mask
from params import EOS_TOKEN, BOS_TOKEN, DATA_PATH


def read_examples_from_file(file_path):
    examples = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            examples.append(line.strip())
    
    return examples


def load_model(word_field, path):
    vocab_size = len(word_field.vocab)
    model = EncoderDecoder(source_vocab_size=vocab_size, target_vocab_size=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()

    return model


def get_tokens_from_words(word_field, string):
    inputs = [BOS_TOKEN] + word_field.preprocess(string) + [EOS_TOKEN]
    return torch.tensor([word_field.vocab.stoi[input] for input in inputs])
    

def get_words_from_tokens(word_field, tokens):
    return [word_field.vocab.itos[token] for token in tokens]


def get_summarization(model, source_input_tokens, word_field):
    target = torch.tensor(word_field.vocab.stoi[BOS_TOKEN]).view(1, 1).to(DEVICE)
    
    while word_field.vocab.stoi[EOS_TOKEN] not in target[0]:
        source_mask, target_mask = make_mask(source_input_tokens, target, pad_idx=1)
        logits = model(source_input_tokens, target,  source_mask, target_mask)
        target = torch.cat((target, torch.argmax(logits[0], dim=1).view(1, -1)), dim=-1)

    return ' '.join(get_words_from_tokens(word_field, target[0]))


def write_summarization(to, examples, model, word_field):
    with open(to, "w", encoding="utf-8") as file:
        for example in examples:
            input = get_words_from_tokens(word_field, example)
            input = filter(lambda x: x != '<pad>', input)

            print(f"input:\n {' '.join(input)}", file=file)
            summarization = get_summarization(model, example.view(1, -1), word_field)
            print(f"output:\n {summarization}\n", file=file)  


def main(model_path, file_path):
    word_field = load_word_field(DATA_PATH)
    model = load_model(word_field, model_path)

    train_dataset = load_dataset(DATA_PATH + "/train/")
    test_dataset = load_dataset(DATA_PATH + "/test/")
    _, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset), batch_sizes=(16, 32), shuffle=True, device=DEVICE, sort=False
    )

    test_examples = next(iter(test_iter)).source.T[0:5]
    write_summarization("test_summarizations.txt", test_examples, model, word_field)

    examples_from_file = read_examples_from_file(file_path)
    preprocessed_inp = [get_tokens_from_words(word_field, inp).to(DEVICE) for inp in examples_from_file]
    write_summarization("our_test_summarization.txt", preprocessed_inp, model, word_field)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="summarization.py")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model.pt",
    )
    parser.add_argument(
        "--file_path",
        type=str,
    )

    args = parser.parse_args()
    main(args.model_path, args.file_path)

