# Task 1: Iris Dataset Visualization
# Objective: Load, explore, and visualize dataset using pandas, seaborn, matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)



print("\n--- Dataset Shape ---")
print(iris.shape)

print("\n--- Column Names ---")
print(iris.columns)

print("\n--- First 5 Rows ---")
print(iris.head())

print("\n--- Dataset Info ---")
iris.info()

print("\n--- Statistical Summary ---")
print(iris.describe())



plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=iris,
    x="sepal_length",
    y="petal_length",
    hue="species"
)

plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")

plt.savefig(os.path.join(output_dir, "scatter_plot.png"))
plt.show()



iris.hist(figsize=(10, 6))
plt.suptitle("Feature Distributions")

plt.savefig(os.path.join(output_dir, "histograms.png"))
plt.show()



plt.figure(figsize=(10, 5))

sns.boxplot(data=iris)

plt.title("Box Plot - Feature Distribution")

plt.savefig(os.path.join(output_dir, "boxplot.png"))
plt.show()

print("\nTask 1 Completed Successfully!")
