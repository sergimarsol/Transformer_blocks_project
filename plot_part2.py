import matplotlib.pyplot as plt

# Data from part2_pretraining_results.txt (seed=42)
iters   = [100,    200,    300,    400,    500]
train_ppl = [573.73, 416.41, 387.98, 227.39, 201.42]

plt.figure(figsize=(8, 5))
plt.plot(iters, train_ppl, 'o-', color='#1f77b4', label='Train Perplexity')
plt.xlabel('Iteration')
plt.ylabel('Perplexity')
plt.title('Part 2: Decoder Pretraining — Training Perplexity')
plt.xticks(iters)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('part2_perplexity.png', dpi=150)
plt.show()
print("Saved to part2_perplexity.png")
