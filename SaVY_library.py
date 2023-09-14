# import libraries
import pandas as pd # for dataframe processing
import matplotlib   # for plots
import matplotlib.pyplot as plt # for plots
import seaborn as sns   # for plots

# displays the distribution of the Class variable
# requirement: class variable should be nominal with 2 values 1 and 0
# inputs:   df - dataframe 
#           class_var_name - class variable
# output:   distribution where 1 is blue and 0 is red
def display_class_distribution_plot(df, class_var_name):
    # Set the style of visualization
    sns.set(style="whitegrid")

    # 1. Plot the frequency of the Class variable
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=class_var_name, data=df, palette={1: "blue", 0: "red"})

    # Annotate each bar with its count
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
        
    plt.title('Distribution of Class Variable')
    plt.show()
