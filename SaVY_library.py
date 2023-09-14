# import libraries
import pandas as pd # for dataframe processing
import matplotlib   # for plots
import matplotlib.pyplot as plt # for plots
import seaborn as sns   # for plots

# Displays the distribution of the Class variable
# Requirement: class variable should be nominal with 2 values 1 and 0
# inputs:   df - dataframe 
#           class_var_name - class variable
#           label0 - what class value 0 means as a string
#           label1 - what class value 1 means as a string
# output:   distribution where 1 is blue and 0 is red
def display_class_distribution(df, class_var_name, label0 = "0", label1 = "1"):
    # Set the style of visualization
    sns.set(style="whitegrid")

    # 1. Plot the frequency of the Class variable
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=class_var_name, data=df, palette={1: "blue", 0: "red"})

    # Check the unique values in the class_var_name column and set x-axis labels accordingly
    unique_vals = df[class_var_name].unique()
    labels = [f'0: {label0}' if val == 0 else f'1: {label1}' for val in unique_vals]
    ax.set_xticklabels(labels)

    # Annotate each bar with its count
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
        
    plt.title(f"Distribution of {class_var_name} Variable (1: {label1}, 0: {label0})")
    plt.show()

    return

# Displays the distribution of the an attribute partitioned by the class variable
# Requirement: class variable should be nominal with 2 values 1 and 0
# inputs:   df - dataframe 
#           class_var_name - class variable name in string
#           attribute_name - attribute name in string
#           label0 - what class value 0 means as a string
#           label1 - what class value 1 means as a string
# output:   distribution the attribute where if class = 1 is blue and class = 0 is red
def display_attribute_v_class_distribution(df, class_var_name, attribute_name, label0 = "0", label1 = "1"):
    # Set the style of visualization
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 6))
    ax = sns.histplot(data=df, x=attribute_name, hue=class_var_name, palette={1: "blue", 0: "red"}, bins=30, kde=False, element="step")
    plt.title(f"Histogram of {attribute_name} based on {class_var_name}")
    plt.xlabel(attribute_name)
    plt.ylabel('Count')

    # Remove the old legend
    ax.legend_.remove()

    # Add a new legend with custom labels
    ax.legend(title=class_var_name, labels=[f'1: {label1}', f'0: {label0}'], loc='upper right')

    plt.show()

    return
