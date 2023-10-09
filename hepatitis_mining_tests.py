# import libraries
import os   # for file system manipulations
import numpy as np  # for arrays and other mathematical operations
import statsmodels.api as sm    # for regressions and statistical models
import pandas as pd # for dataframe processing
import matplotlib   # for plots
import matplotlib.pyplot as plt # for plots
import seaborn as sns   # for plots

# all sklearn imports
from sklearn.preprocessing import StandardScaler    # for scaling values
from sklearn.decomposition import PCA   # for principal components
from sklearn.impute import SimpleImputer    # to impute values
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.feature_selection import SelectFromModel, RFE   # for feature extraction
from sklearn.tree import DecisionTreeClassifier    # for decision tree classifier
from sklearn.linear_model import LassoCV    # for Lasso cross-validation regression model

# import custom library
from SaVY_library import \
    display_class_distribution, \
    display_attribute_v_class_distribution

plots_on = False

# load hepatitis data
df = pd.read_csv("C:/Users/vladc/OneDrive/Documents/GMU MS Health Informatics Courses/HAP 780 Data Mining in Health Care/Week 3/Assignments/data/GMU_HAP780_Wk3_LM4_hepatitis(1).csv")

# SsECTION 1: ATTRIBUTE MODIFICATION AND DISTRIBUTION
# remap the Class variable to died is 1 and lived is 0 to make is nominal and binary
df['Class'] = df['Class'].map({1: 1, 2: 0})

# Strip the whitespace from the column names
df.columns = [col.strip() for col in df.columns]

if plots_on:
    # 1. Plot the frequency of the Class variable
    display_class_distribution(df, 'Class', label0="Lived", label1="Died")

    # 2. Plot separate distributions for the ALBUMIN variable based on the Class
    display_attribute_v_class_distribution(df, 'Class', 'ALBUMIN', label0="Lived", label1="Died")


# SECTION 2: FINDING PRINCIPAL COMPONENTS (SIMPLE)
# 1. Define features variable
features = df.drop(columns=['Class'])

# 2. Preprocessing

# 2.a. Use mean imputation to fill missing values
# There are many imputation techniques, we will just use the mean in this example
imputer = SimpleImputer(strategy="mean")
imputed_features = imputer.fit_transform(features)

# 2.b. Extracting numerical features and scaling them
# It's a good practice to scale the data before performing PCA because PCA is sensitive to the magnitudes of the features.
scaled_features = StandardScaler().fit_transform(imputed_features)

# 3. Applying PCA for 2 components as an example
# You can decide the number of components you want, but for visualization purposes, people usually go for 2 or 3 components.
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

# 4. Display components
loadings = pca.components_

# Open the file for writing
with open("./results/pca_2_comp.txt", "w") as file:
    # Write the loadings for all principal components to the file
    for i, component in enumerate(loadings):
        file.write(f"\nLoadings for principal component {i+1}:\n")
        for feature, loading in zip(features.columns, component):
            file.write(f"{feature}: {loading:.4f}\n")

# Convert the principal components to a DataFrame for easier plotting
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pc_df['Class'] = df['Class']

if plots_on:
    # 4. Visualization
    # If you choose 2 components, you can visualize them in a 2D scatter plot, and for 3 components, you can visualize in 3D.
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='PC1', y='PC2', hue='Class', data=pc_df, palette={1: "blue", 0: "red"})
    plt.title('Principal Component Analysis (2 Components)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# SECTION 3: FINDING PRINCIPAL COMPONENTS (RANKED)
# 1. Preprocess the data
features = df.drop(columns=['Class'])
imputer = SimpleImputer(strategy="mean")
imputed_features = imputer.fit_transform(features)
scaled_features = StandardScaler().fit_transform(imputed_features)

# 2. Apply PCA without reducing dimensionality to capture all variance
pca = PCA()
pca.fit(scaled_features)

# 3 Display components
loadings = pca.components_

# Open the file for writing
with open("./results/pca_ranked.txt", "w") as file:
    # Write the loadings for all principal components to the file
    for i, component in enumerate(loadings):
        file.write(f"\nLoadings for principal component {i+1}:\n")
        for feature, loading in zip(features.columns, component):
            file.write(f"{feature}: {loading:.4f}\n")

if plots_on:
    # 4. Visualize the explained variance for each component
    explained_var = pca.explained_variance_ratio_

    plt.figure(figsize=(10,6))
    plt.bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, label='Individual Explained Variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.show()

# SECTION 4: ATTRIBUTE SELECTION
# 1. Random Forest Classifier Feature Selection
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(imputed_features, df['Class'])

# Extracting important features based on an importance threshold
selector = SelectFromModel(clf, prefit=True)
selected_features_idx = selector.get_support()
selected_feature_names = features.columns[selected_features_idx]

# Print some summaries
print("\nATTRIBUTE SELECTION: Model Parameters for Random Forest:")
print(clf.get_params())

print("Selected Features based on Random Forest Classifier Importance:")
print(selected_feature_names)

# Pair feature names with their importance scores
feature_importances = list(zip(features.columns, clf.feature_importances_))

# Sort the feature importances in descending order
sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# Display sorted features
for name, importance in sorted_feature_importances:
    print(f"{name}: {importance:.4f}")

# 2. Information Gain Attribute Selection
# Fit a decision tree to the data
tree = DecisionTreeClassifier(criterion='entropy')  # Using entropy as the split criterion
tree.fit(imputed_features, df['Class'])

# Extract and sort feature importances
feature_importances = list(zip(features.columns, tree.feature_importances_))
sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# Select features based on a threshold or the top N features
selector = SelectFromModel(tree, prefit=True, threshold=0.05)  # For example, threshold set to 0.05
selected_features = selector.transform(imputed_features)

# Determine which features were selected
selected_mask = selector.get_support()
selected_names = features.columns[selected_mask]

# Print some summaries
# Print model parameters
print("\nATTRIBUTE SELECTION: Model Parameters for DecisionTree using Information Gain (entropy):")
print(tree.get_params())

# Display sorted features
print("\nFeatures ranked by information gain:")
for name, importance in sorted_feature_importances:
    print(f"{name}: {importance:.4f}")

# Print the selected features
print("\nSelected features based on information gain threshold:")
for name in selected_names:
    print(name)

# 3. Correlation-based Feature Selection (CFS) is not available

# 4. Wrapper Based: Classifier Subset Evaluator (CSE) or Recursive Feature Elimination (RFE) method
# Feature selection using RFE
rfe = RFE(estimator=clf, n_features_to_select=10)  # Here, we want the top 10 features
fit = rfe.fit(imputed_features, df['Class'])

# Get the selected features
selected_features = features.columns[fit.support_]

print("ATTRIBUTE SELECTION: Wrapper Based: Classifier Subset Evaluator (CSE) or Recursive Feature Elimination (RFE) RandomForest Classifier Parameters:")
print(clf.get_params())

print("\nTop features selected by Classifier Subset Evaluator using RandomForest:")
for feature in selected_features:
    print(feature)

# 5. Principal Components Attribute Selection
pca = PCA()
principal_components = pca.fit_transform(scaled_features)

# Print cumulative explained variance
explained_variance = pca.explained_variance_ratio_.cumsum()
print("\nATTRIBUTE SELECTION: Cumulative Explained Variance by Principal Components:")
for i, var in enumerate(explained_variance, 1):
    print(f"PC{i}: {var:.4f}")

# Select top N components based on a threshold or your preference
# For example, let's say we want to keep enough components to explain 95% of the variance
n_components = np.argmax(explained_variance >= 0.95) + 1
selected_components = principal_components[:, :n_components]

print(f"\nThe top {n_components} principal components that explain 95% or more of the variance are:")
for i in range(n_components):
    print(f"Principal Component {i + 1}:")
    print(selected_components[:, i])
    print()

# 6 Filter method using Peason correlation
# Get Pearson Correlation
cor=df.corr()

if plots_on:
    # Plot heatmap of Pearson Correlation
    plt.figure(figsize=(12,12))
    sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
    plt.show()

# Correlate with Class variable
cor_target = abs(cor["Class"])

# Select highly correlated features (greater than 0.5)
selected_features = cor_target[cor_target>0.5]
print("\nATTRIBUTE SELECTION: Filter Method Pearson Correlation Relevant_Features:\n", selected_features)

# 6. Backward Elimination
# We include all variables then remove the variable with the highest non-significant p-value

alpha = 0.05 # significance level

cols = list(features.columns)   # list of columns
pmax = 1    # initialize maximum p-value

while(len(cols)>0):
    p=[]    # initialize p-values list

    # Execute ordinary least squares (OLS) with remaining columns
    X_1 = features[cols]
    X_1 = X_1.fillna(X_1.mean())    # fill missing values with mean
    X_1 = sm.add_constant(X_1)    
    model = sm.OLS(df["Class"], X_1).fit()

    # get p-values from the model and get the max p-value
    p = pd.Series(model.pvalues.values[1:], index = cols)
    pmax = max(p)

    # get the feature with the max p-value
    feature_w_max_p = p.idxmax()

    # drop feature if maximum p-value non-significant, stop if not
    if(pmax > alpha):
        cols.remove(feature_w_max_p)
    else:
        break

# Select attributes
selected_features = cols
print("\nATTRIBUTE SELECTION: Backward Elimination Using Ordinary Least Squares:\n", cols)

# 7. Embedded Method LASSO Cross-Validation
reg = LassoCV()
reg.fit(imputed_features, df["Class"])

# Get the feature names
feature_names = features.columns

# Filter and display features with non-zero coefficients
non_zero_features = feature_names[reg.coef_ != 0]

print("\nATTRIBUTE SELECTION: Embedded Method LASSO Cross-Validation")
print("\nBest alpha using built-in LassoCV: %f" %reg.alpha_)
print("\nBest score using built-in LassoCV: %f" %reg.score(imputed_features, df["Class"]))
print("\nThe selected attributes with non-zero coefficients:")
for feature in non_zero_features:
    print(feature)

if plots_on:
    coef = pd.Series(reg.coef_, index = features.columns)
    imp_coef = coef.sort_values()
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using Lasso Model")
    plt.show()

# SECTION 5: RESAMPLING
# 1. Random resampling without replacement
sampled_data = df.sample(frac=0.5, replace=False)    # assume sample size is 50%

if plots_on:
    # reset index before plotting
    sampled_data = sampled_data.reset_index(drop=True)

    # a. Plot the frequency of the Class variable
    display_class_distribution(sampled_data, 'Class', label0="Lived", label1="Died")

    # b. Plot separate distributions for the ALBUMIN variable based on the Class
    display_attribute_v_class_distribution(sampled_data, 'Class', 'ALBUMIN', label0="Lived", label1="Died")

# 2. Random resampling with replacement
sampled_data = df.sample(frac=0.5, replace=True)   # assume sample size is 50%

# reset index before plotting
sampled_data = sampled_data.reset_index(drop=True)

if plots_on:
    # reset index before plotting
    sampled_data = sampled_data.reset_index(drop=True)

    # a. Plot the frequency of the Class variable
    display_class_distribution(sampled_data, 'Class', label0="Lived", label1="Died")

    # b. Plot separate distributions for the ALBUMIN variable based on the Class
    display_attribute_v_class_distribution(sampled_data, 'Class', 'ALBUMIN', label0="Lived", label1="Died")

# 3. Resampling to balance outcome variable without replacement
# Split data based on classes
class_0 = df[df['Class'] == 0]
class_1 = df[df['Class'] == 1]

# Sample data from the majority class without replacement to match the minority class's size
if len(class_1) < len(class_0):
    class_0_downsampled = class_0.sample(len(class_1), replace=False)

    # Combine downsampled majority class with minority class
    sampled_data = pd.concat([class_0_downsampled, class_1])
else:
    class_1_downsampled = class_1.sample(len(class_0), replace=False)

    # Combine downsampled majority class with minority class
    sampled_data = pd.concat([class_1_downsampled, class_0])

if plots_on:
    # reset index before plotting
    sampled_data = sampled_data.reset_index(drop=True)

    # a. Plot the frequency of the Class variable
    display_class_distribution(sampled_data, 'Class', label0="Lived", label1="Died")

    # b. Plot separate distributions for the ALBUMIN variable based on the Class
    display_attribute_v_class_distribution(sampled_data, 'Class', 'ALBUMIN', label0="Lived", label1="Died")

# 4. Resampling to balance outcome variable without replacement match original sample size
# Calculate the number of samples to take to balance the classes and match the original dataset size
num_samples = len(df) // 2  # Assuming two classes for simplicity

# Sample data from each class with replacement
class_0_upsampled = class_0.sample(num_samples, replace=True)
class_1_upsampled = class_1.sample(num_samples, replace=True)

# Combine upsampled classes
sampled_data = pd.concat([class_0_upsampled, class_1_upsampled])

if plots_on:
    # reset index before plotting
    sampled_data = sampled_data.reset_index(drop=True)

    # a. Plot the frequency of the Class variable
    display_class_distribution(sampled_data, 'Class', label0="Lived", label1="Died")

    # b. Plot separate distributions for the ALBUMIN variable based on the Class
    display_attribute_v_class_distribution(sampled_data, 'Class', 'ALBUMIN', label0="Lived", label1="Died")