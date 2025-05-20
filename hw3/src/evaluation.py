from model import EncoderDecoder
from utilities import convert_batch, make_mask, load_dataset, load_word_field, DEVICE
from summarization import get_words_from_tokens
from params import BOS_TOKEN, EOS_TOKEN
from torchtext.data import BucketIterator
import torch
import json
import evaluate


def generate_sequence(model, word_field, source_input, target_input, source_mask, target_mask, max_len):
    model.eval()
    with torch.no_grad():
        bos_idx = word_field.vocab.stoi[BOS_TOKEN]
        eos_idx = word_field.vocab.stoi[EOS_TOKEN]
        encoder_output = model.encoder(source_input, source_mask)
        for j in range(max_len):
            source_mask, target_mask = make_mask(source_input, target_input, pad_idx=1)
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

def ROUGE_evaluation(source_vocab_size, target_vocab_size, path_to_model, word_field, iter, with_embs):
    if (with_embs):
        model = EncoderDecoder(source_vocab_size, target_vocab_size, True)
    else:
        model = EncoderDecoder(source_vocab_size, target_vocab_size, False)
    model.load_state_dict(torch.load(path_to_model, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for batch in iter:
            source_input, target_input, source_mask, target_mask = convert_batch(batch)
            source_input = source_input.to(DEVICE)
            source_mask = source_mask.to(DEVICE) 
            for i in range(target_input.shape[0]):
                ref_text = ' '.join(get_words_from_tokens(word_field, target_input[i]))  
                references.append(ref_text)
            prediction_tokens = generate_sequence(model, word_field, source_input, target_input, source_mask, target_mask, target_input.shape[1] // 2)
            predictions.append(prediction_tokens)
    
    rouge = evaluate.load('rouge')  
    results = rouge.compute(predictions=predictions, references=references)
    print(results)

    with open('hw3/src/evaluation_out/rouge_results_with_embs.json', 'w') as f:
        json.dump(results, f, indent=2)

def main():
    test_dataset = load_dataset("./data/test")
    train_dataset = load_dataset("./data/train") 
    test_iter = BucketIterator(
        test_dataset, 
        batch_size=1, 
        device=DEVICE
    )
    
    train_iter = BucketIterator(
        train_dataset, 
        batch_size=1, 
        device=DEVICE
    )
    with open('dataset_stats.json', 'r') as f:
        data_stats = json.load(f)

    word_field = load_word_field("./data") 
    vocab_size =  data_stats["vocab_size"]
     
    ROUGE_evaluation(
        source_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        path_to_model="./models/model_with_embs.pt",
        word_field=word_field,
        iter=test_iter,
        with_embs = True
    )

if __name__ == '__main__':
    main()