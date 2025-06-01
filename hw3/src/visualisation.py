from utilities import convert_batch
from summarization import get_words_from_tokens
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention, x_tokens, y_tokens, title="", font_scale=0.6):
    plt.figure(figsize=(30, 30))
    sns.set_theme(font_scale=font_scale)
    
    ax = sns.heatmap(
        attention,
        annot=False,
        fmt=".2f",
        xticklabels=x_tokens,
        yticklabels=y_tokens,
        cmap="YlGnBu",
        cbar=False
    )
    
    plt.title(title, pad=20)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    # plt.show()


def visualise(model, word_field, batch):
    words = []
    source_input, target_input, source_mask, target_mask = convert_batch(batch)
    words = get_words_from_tokens(word_field, batch.source) 
    blocks = len(model.encoder._blocks)
    heads = model.encoder._blocks[0]._self_attn._attn_probs.shape[1]
    encoder_output = model.encoder(source_input, source_mask)
    for block in range(blocks):
        for head in range(heads):
            attention = model.encoder._blocks[block]._self_attn._attn_probs[0, head].cpu().detach().numpy()
            plot_attention(
                attention, 
                words, 
                words,
            title=f"Block {block+1}, Head {head+1}"
            )
            plt.savefig(f"hw3/visualization_out/example_3/block_{block + 1}/head_{head+1}.jpg", bbox_inches='tight', dpi=300)
            plt.close() 
