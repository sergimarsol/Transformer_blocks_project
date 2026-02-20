
import matplotlib.pyplot as plt
import torch

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, prefix="", is_causal=False):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)
        num_real_tokens = min(len(wordids), block_size)

        # Get the word labels for the axes (only real tokens)
        words = self.tokenizer.decode(wordids[:num_real_tokens]).split()

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        # Move input to the same device as the model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the model
        self.model.eval()
        with torch.no_grad():
            _, attn_maps = self.model(input_tensor)
        self.model.train()

        # Display the number of attention maps
        print("Number of attention maps (layers):", len(attn_maps))

        # attn_maps[j] has shape (1, num_heads, T, T)
        num_heads = attn_maps[0].size(1)
        print(f"Number of heads per layer: {num_heads}")

        # Visualize and save attention maps – one plot per (layer, head)
        for j, attn_map in enumerate(attn_maps):
            # attn_map: (1, num_heads, T, T)
            attn_map_cpu = attn_map.squeeze(0).detach().cpu()  # (num_heads, T, T)

            for h in range(num_heads):
                head_attn = attn_map_cpu[h]  # (T, T)

                # Check if the attention probabilities sum to 1 over rows
                total_prob_over_rows = head_attn.sum(dim=-1)
                if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                    print(f"Layer {j+1}, Head {h+1}: FAILED normalization test")
                    print("  Row sums:", total_prob_over_rows.numpy())
                else:
                    print(f"Layer {j+1}, Head {h+1}: Passed normalization test (rows sum to 1)")

                # For causal models: verify upper triangle is zero
                if is_causal:
                    upper = torch.triu(head_attn[:num_real_tokens, :num_real_tokens], diagonal=1)
                    max_upper = upper.max().item()
                    if max_upper < 1e-6:
                        print(f"Layer {j+1}, Head {h+1}: Passed causal mask test (upper triangle ≈ 0, max={max_upper:.2e})")
                    else:
                        print(f"Layer {j+1}, Head {h+1}: FAILED causal mask test (upper triangle max={max_upper:.4f})")

                # Crop to real tokens only for clearer visualization
                head_attn_np = head_attn[:num_real_tokens, :num_real_tokens].numpy()

                # Create a heatmap of the attention map
                fig, ax = plt.subplots(figsize=(8, 6))
                cax = ax.imshow(head_attn_np, cmap='hot', interpolation='nearest')
                ax.set_xticks(range(num_real_tokens))
                ax.set_yticks(range(num_real_tokens))
                ax.set_xticklabels(words, rotation=90, fontsize=8)
                ax.set_yticklabels(words, fontsize=8)
                ax.xaxis.tick_top()
                ax.set_xlabel("Key")
                ax.set_ylabel("Query")
                fig.colorbar(cax, ax=ax)
                model_label = prefix if prefix else ""
                plt.title(f"{model_label}Layer {j+1}, Head {h+1}")
                plt.tight_layout()

                # Save the plot
                fname = f"{prefix}attention_map_layer{j+1}_head{h+1}.png"
                plt.savefig(fname, dpi=150)

                # Show the plot
                plt.show()


