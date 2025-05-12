from torchtext.data import Field, Example, Dataset
import pandas as pd
from tqdm.auto import tqdm

from utilities import save_dataset, save_word_field
from params import DATA_PATH, BOS_TOKEN, EOS_TOKEN


def main():
    word_field = Field(tokenize='moses', init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
    fields = [('source', word_field), ('target', word_field)]
    
    data = pd.read_csv('/home/chivoro/Desktop/ml_hw/hw3/src/news.csv', delimiter=',')

    examples = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        source_text = word_field.preprocess(row.text)
        target_text = word_field.preprocess(row.title)
        examples.append(Example.fromlist([source_text, target_text], fields))
        
    dataset = Dataset(examples, fields)
    
    train_dataset, test_dataset = dataset.split(split_ratio=0.85)
    
    print('Train size =', len(train_dataset))
    print('Test size =', len(test_dataset))

    word_field.build_vocab(train_dataset, min_freq=7)
    print('Vocab size =', len(word_field.vocab))
    
    save_dataset(train_dataset, DATA_PATH + "/train/") 
    save_dataset(test_dataset, DATA_PATH + "/test/") 
    
    save_word_field(word_field, DATA_PATH)


if __name__ == "__main__":
    main()
