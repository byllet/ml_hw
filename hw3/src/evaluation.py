# Задание 2. Оценка моделей с помощью ROUGE-метрики
from model import EncoderDecoder
from utilities import convert_batch, make_mask, load_dataset, load_word_field
from summarization import get_words_from_tokens # TODO: Fix it!!
from params import BOS_TOKEN, EOS_TOKEN
from torchtext.data import BucketIterator
import torch


PAD_IDX = 1  

def generate_sequence(model, word_field, source_input, max_len):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        encoder_output = model.encoder(source_input, source_mask)
        bos_idx = word_field.vocab.stoi[BOS_TOKEN]
        eos_idx = word_field.vocab.stoi[EOS_TOKEN]
        target_input = torch.tensor([[bos_idx]])
        target_input.to(DEVICE)
        for i in range(max_len):
            source_mask, target_mask = make_mask(source_input, target_input, pad_idx=PAD_IDX)
            decoder_output = model.decoder(target_input, encoder_output, source_mask, target_mask)            
            next_token_scores = decoder_output[:, -1, :]              
            next_token = torch.argmax(next_token_scores, dim=-1).unsqueeze(1) 
            target_input = torch.cat((target_input, next_token), dim=1)           
            if next_token.item() == eos_idx:
                break        
        generated_tokens = target_input[0].tolist()
        if generated_tokens[0] == bos_idx:
            generated_tokens = generated_tokens[1:]
        if eos_idx in generated_tokens:
            generated_tokens = generated_tokens[:generated_tokens.index(eos_idx)]
        
        generated_text = ' '.join(get_words_from_tokens(word_field, generated_tokens)) 
        return generated_text

def ROUGE_evaluation(source_vocab_size, target_vocab_size, path_to_model, word_field, iter):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoder(source_vocab_size, target_vocab_size, d_model=300, heads_count=15) # тут что-то странное
    model.load_state_dict(torch.load(path_to_model, map_location=DEVICE))
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for batch in iter:
            source_input, input_for_reference, source_mask, target_mask = convert_batch(batch) 
            source_input = source_input.to(DEVICE)
            
            for i in range(input_for_reference.shape[0]): 
                ref_text = ' '.join(get_words_from_tokens(word_field, input_for_reference[i].cpu()))  
                references.append(ref_text)

            prediction_tokens = generate_sequence(model, word_field, source_input, input_for_reference.shape[1] // 2)
            predictions.append(prediction_tokens)
    
    rouge = evaluate.load('rouge')  
    results = rouge.compute(predictions=predictions, references=references)
    print(results)

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    test_dataset = load_dataset("/home/chivoro/Desktop/ml_hw/hw3/data/test") # !!!
    train_dataset = load_dataset("/home/chivoro/Desktop/ml_hw/hw3/data/train") # !!!
    
    test_iter = BucketIterator(
        test_dataset, 
        batch_size=32, 
        device=DEVICE, 
        sort=False,          # Добавлено
        sort_within_batch=False  # Добавлено
    )
    
    train_iter = BucketIterator(
        train_dataset, 
        batch_size=32, 
        device=DEVICE, 
        sort=False,          # Добавлено
        sort_within_batch=False  # Добавлено
    )
    
    word_field = load_word_field("/home/chivoro/Desktop/ml_hw/hw3/data/") # !!!
    vocab_size = 55784 # !!
     
    ROUGE_evaluation(
        source_vocab_size=vocab_size,  # Исправлен порядок параметров
        target_vocab_size=vocab_size,
        path_to_model="/home/chivoro/Desktop/ml_hw/hw3/src/model_epoch.pt",
        word_field=word_field,
        iter=test_iter
    )


if __name__ == '__main__':
    main()