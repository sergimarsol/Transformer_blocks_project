import matplotlib.pyplot as plt
import numpy as np

# Data from part3 exploration summaries (seed=42, best config per method)
methods = ['Full\nAttention', 'Window\n(W=5)', 'Window\n(W=3)', 'Block Sparse\n(B=8)', 'ALiBi']

obama  = [395.51, 377.81, 371.15, 393.90, 354.39]
wbush  = [488.96, 454.91, 469.33, 475.48, 450.14]
hbush  = [430.65, 399.03, 405.88, 425.53, 385.70]
avg    = [(o + w + h) / 3 for o, w, h in zip(obama, wbush, hbush)]

x = np.arange(len(methods))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 1.5 * width, obama, width, label='Obama',   color='#1f77b4')
ax.bar(x - 0.5 * width, wbush, width, label='W. Bush', color='#ff7f0e')
ax.bar(x + 0.5 * width, hbush, width, label='H. Bush', color='#2ca02c')
ax.bar(x + 1.5 * width, avg,   width, label='Average', color='#d62728', edgecolor='black', linewidth=0.8)

ax.set_xlabel('Attention Method')
ax.set_ylabel('Test Perplexity')
ax.set_title('Part 3: Test Perplexity Comparison Across Attention Variants')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(300, 520)

# Add value labels on the average bars
for i, v in enumerate(avg):
    ax.text(x[i] + 1.5 * width, v + 3, f'{v:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('part3_comparison.png', dpi=150)
plt.show()
print("Saved to part3_comparison.png")
