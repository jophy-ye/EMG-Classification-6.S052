import matplotlib.pyplot as plt
import numpy as np

accuracy = [0.4545, 0.4848, 0.5455, 0.5455, 0.5152, 0.5455, 0.5455, 0.6061, 0.5758, 0.6364, 0.6364, 0.6667, 0.5758, 0.5758, 0.5758, 0.5455, 0.5455, 0.5455, 0.5455, 0.5455, 0.5758, 0.5758, 0.6061, 0.6970, 0.5758, 0.5758, 0.5758, 0.5758, 0.5758]
epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 999, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 9999]

# plt.plot(epochs, accuracy, marker='o', linestyle='-')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs Epochs')
# plt.grid(True)
# plt.show()

# epoch 7000
#accuracy 69.7%

predicted_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1]
true_labels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1]

TP = 0 #true positive
FP = 0 #false positive
TN = 0 #true negative
FN = 0 #false negative

for predicted, true in zip(predicted_labels, true_labels):
    if predicted == 1 and true == 1:
        TP += 1
    elif predicted == 0 and true == 1:
        FN += 1
    elif predicted == 1 and true == 0:
        FP += 1
    else:
        TN += 1

# Calculate probabilities
total_samples = len(true_labels)
probabilities = [TP / total_samples, FP / total_samples, TN / total_samples, FN / total_samples]

# Calculate entropy
entropy = 0
for prob in probabilities:
    if prob != 0:  # Avoid taking logarithm of zero
        entropy -= prob * np.log(prob)

print("TP, FP, TN, FN counts:", [TP, FP, TN, FN])
print("TP, FP, TN, FN probabilities:", probabilities)
print("Entropy:", entropy)