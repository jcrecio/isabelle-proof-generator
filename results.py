import numpy as np
import matplotlib.pyplot as plt

# Set the backend to Agg explicitly
plt.switch_backend("Agg")

# Data setup
models = ["Mathstral 7B", "Deepseek 32B", "Deepseek Math 7B"]
x = np.arange(len(models))
width = 0.25  # Width of the bars

# Sample data (replace with actual percentages)
base_scores = [1, 11, 14]  # Base model scores
finetuned_scores = [2, 19, 23]  # Finetuned model scores
rag_scores = [1, 23, 27]  # RAG model scores

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Create bars
rects1 = ax.bar(x - width, base_scores, width, label="Base", color="blue")
rects2 = ax.bar(x, finetuned_scores, width, label="Finetuned", color="green")
rects3 = ax.bar(x + width, rag_scores, width, label="RAG", color="grey")

# Customize the plot
ax.set_ylabel("Performance (%)")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()


# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Adjust layout and save to file
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
