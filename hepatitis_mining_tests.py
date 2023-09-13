# import libraries
import pandas as pd # for dataframe processing
import matplotlib.pyplot as plt # for plots
import seaborn as sns   # for plots
from sklearn.preprocessing import StandardScaler    # for scaling values
from sklearn.decomposition import PCA   # for principal components
from sklearn.impute import SimpleImputer    # to impute values


# load hepatitis data
df = pd.read_csv("C:/Users/vladc/OneDrive/Documents/GMU MS Health Informatics Courses/HAP 780 Data Mining in Health Care/Week 3/Assignments/data/GMU_HAP780_Wk3_LM4_hepatitis(1).csv")


# SECTION 1: ATTRIBUTE MODIFICATION AND DISTRIBUTION
# remap the Class variable to died is 1 and lived is 0 to make is nominal and binary
df['Class'] = df['Class'].map({1: 1, 2: 0})

# Strip the whitespace from the column names
df.columns = [col.strip() for col in df.columns]

# Set the style of visualization
sns.set(style="whitegrid")

# 1. Plot the frequency of the Class variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette={1: "blue", 0: "red"})
plt.title('Distribution of Class Variable (1: Died, 0: Lived)')
plt.show()

# 2. Plot separate distributions for the ALBUMIN variable based on the Class (density plot)
plt.figure(figsize=(8, 6))
sns.kdeplot(df[df['Class'] == 1]['ALBUMIN'], fill=True, color="blue", label="Died")
sns.kdeplot(df[df['Class'] == 0]['ALBUMIN'], fill=True, color="red", label="Lived")
plt.title('Distribution of ALBUMIN based on Class')
plt.xlabel('ALBUMIN')
plt.ylabel('Density')
plt.legend()
plt.show()

# 3 Plot separate histograms for the ALBUMIN variable based on the Class
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='ALBUMIN', hue='Class', palette={1: "blue", 0: "red"}, bins=30, kde=False, element="step")
plt.title('Histogram of ALBUMIN based on Class')
plt.xlabel('ALBUMIN')
plt.ylabel('Count')
plt.show()

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

# Convert the principal components to a DataFrame for easier plotting
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pc_df['Class'] = df['Class']

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

# 3. Visualize the explained variance for each component
explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(10,6))
plt.bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, label='Individual Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.show()

# SECTION 4: ATTRIBUTE SELECTION