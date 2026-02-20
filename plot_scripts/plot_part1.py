import matplotlib.pyplot as plt

# Data from part1_results.txt (seed=42)
epochs = list(range(1, 16))
train_acc = [44.17, 49.62, 58.17, 65.68, 71.51, 80.02, 85.71, 89.77,
             92.40, 92.59, 95.36, 97.32, 97.90, 97.18, 97.66]
test_acc  = [45.33, 50.53, 58.67, 64.00, 73.20, 77.33, 78.27, 81.73,
             81.73, 79.33, 85.07, 85.60, 84.53, 83.87, 83.33]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, 'o-', label='Train Accuracy', color='#1f77b4')
plt.plot(epochs, test_acc,  's--', label='Test Accuracy',  color='#ff7f0e')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Part 1: Encoder + Classifier — Training & Test Accuracy')
plt.xticks(epochs)
plt.ylim(40, 100)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('part1_accuracy.png', dpi=150)
plt.show()
print("Saved to part1_accuracy.png")
