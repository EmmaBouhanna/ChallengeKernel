import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_train(acc_list):
    sns.set_style("dark")
    plt.figure(figsize=(8,8))
    plt.plot(acc_list, label="Validation Accuracy")
    plt.ylabel("Accuracy %")
    plt.xlabel("# Epochs")
    plt.legend()
    plt.show()
