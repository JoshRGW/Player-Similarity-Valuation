#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist


# # 2. Data Preprocessing

# In[2]:


# Load the data
data = pd.read_csv('fm_male_players.csv', low_memory=False)


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.duplicated().sum()


# In[8]:


data[data.duplicated()]


# In[9]:


# Plotting the distribution of the 'overall' column
plt.figure(figsize=(10, 6))
sns.histplot(data['overall'], bins=20, kde=True)
plt.title('Distribution of Overall Ratings')
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[10]:


league_counts = data['leaguename'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=league_counts.index, y=league_counts.values, palette="Set2")
plt.title('Number of Players in Different Leagues')
plt.xlabel('League Name')
plt.ylabel('Number of Players')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[11]:


# List of columns to drop
columns_to_drop = [
    'clubjerseynumber', 'nationteamid', 'nationposition', 
    'nationjerseynumber', 'bodytype', 'leaguelevel', 'ls', 'st', 'rs', 'lw', 'lf', 
    'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 
    'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk'
]

# Drop the specified columns
data = data.drop(columns=columns_to_drop)

# Verify the columns have been dropped
print(data.columns)


# In[12]:


# Check for missing values
missing_values = data.isnull().sum()

# Display the columns with the most missing values
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(ascending=False)


# In[13]:


# Drop columns with too many missing values and non-critical for similarity analysis
columns_to_drop = ['playertags', 'playertraits', 'clubteamid', 'leagueid', 'leaguename', 
                   'clubname', 'goalkeepingspeed']
data.drop(columns=columns_to_drop, inplace=True)

# Verify the columns have been dropped
print(data.columns)


# In[14]:


# Impute missing values in numeric columns with median values
numeric_cols = ['shooting', 'physic', 'defending', 'dribbling', 
                'passing', 'pace', 'mentalitycomposure']
for col in numeric_cols:
    data[col].fillna(data[col].median(), inplace=True)


# In[15]:


# For categorical columns, impute missing values with a placeholder
categorical_cols = ['clubposition', 'preferredfoot', 'workrate']
for col in categorical_cols:
    data[col].fillna('Unknown', inplace=True)
    
# Confirm no missing values remain
print(data.isnull().sum())


# # 3. K-Means Clustering Model

# In[16]:


# Feature Selection: Select relevant features for kMEANS Clustering
selected_features = ['playerid', 'shortname', 'longname', 'overall', 'clubposition', 
                     'potential', 'age', 'heightcm', 'preferredfoot', 'weakfoot', 'skillmoves', 
                     'workrate', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 
                     'attackingfinishing', 'skilldribbling', 'skilllongpassing', 'skillballcontrol', 
                     'movementacceleration', 'movementsprintspeed', 'movementreactions', 'movementbalance', 
                     'powerstamina', 'powerstrength', 'mentalityaggression', 'mentalityinterceptions', 
                     'mentalitypositioning', 'mentalityvision', 'mentalitycomposure', 'defendingmarkingawareness',
                     'defendingstandingtackle', 'defendingslidingtackle', 'goalkeepingdiving', 
                     'goalkeepinghandling', 'goalkeepingkicking', 'goalkeepingpositioning', 'goalkeepingreflexes'
]


# In[17]:


# Extract the relevant data
similarity_data = data[selected_features]


# In[18]:


# One-hot encode categorical features
similarity_data = pd.get_dummies(similarity_data, columns=['preferredfoot', 'workrate', 'clubposition'])


# In[19]:


# Display the prepared data for similarity analysis
print(similarity_data.head())


# In[20]:


# Drop non-numeric and id columns for clustering
clustering_data = similarity_data.drop(columns=['playerid', 'shortname', 'longname'])


# In[21]:


# Standardise the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)


# In[22]:


# Apply the elbow method to find the optimal number of clusters
inertia = []
range_clusters = range(1, 11)
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)


# In[23]:


# Plot the elbow graph
plt.figure(figsize=(8, 4))
plt.plot(range_clusters, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


# In[24]:


# Number of clusters from the elbow method analysis
optimal_clusters = 3

# Apply KMeans with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=0)
data['cluster_optimal'] = kmeans_optimal.fit_predict(scaled_data)

# Ensure only numeric data is included for cluster summary
numeric_columns = ['overall', 'potential', 'age', 'heightcm', 'weakfoot', 'skillmoves', 'pace', 'shooting', 
                   'passing', 'dribbling', 'defending', 'physic', 'attackingfinishing', 'skilldribbling', 
                   'skilllongpassing', 'skillballcontrol', 'movementacceleration', 'movementsprintspeed', 
                   'movementagility', 'movementreactions', 'movementbalance', 'powershotpower', 
                   'powerjumping', 'powerstamina', 'powerstrength', 'powerlongshots', 'mentalityaggression', 
                   'mentalityinterceptions', 'mentalitypositioning', 'mentalityvision', 'mentalitycomposure', 
                   'defendingmarkingawareness', 'defendingstandingtackle', 'defendingslidingtackle', 
                   'goalkeepingdiving', 'goalkeepinghandling', 'goalkeepingkicking', 'goalkeepingpositioning', 
                   'goalkeepingreflexes']

cluster_summary = data.groupby('cluster_optimal')[numeric_columns].mean()


# In[25]:


# Plotting bar charts for each attribute with numerical labels
for attribute in numeric_columns:
    plt.figure(figsize=(8, 4))
    barplot = sns.barplot(x=cluster_summary.index, y=cluster_summary[attribute])
    plt.title(f'Average {attribute} per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(f'Average {attribute}')

    # Add numerical labels on the bars
    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.2f}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 10), 
                         textcoords='offset points')
    
    plt.show()


# In[26]:


# These attributes will be selected for the heatmap
selected_attributes = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'cluster_optimal']

# Compute the correlation matrix only on the selected attributes (excluding 'cluster_optimal')
correlation_matrix = data[selected_attributes[:-1]].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Selected Attributes')
plt.show()


# # 3.1 Dataframes for Clusters

# In[27]:


#sepeate each cluters into a seperate dataframe
data0 = data[data['cluster_optimal'] == 0]
data1 = data[data['cluster_optimal'] == 1]
data2 = data[data['cluster_optimal'] == 2]


# In[28]:


data0


# In[29]:


data1


# In[30]:


data2


# In[31]:


# Function to compute statistics for each cluster
def compute_cluster_statistics(cluster_data, cluster_name):
    stats = cluster_data[numeric_columns].describe().transpose()
    stats['median'] = cluster_data[numeric_columns].median()
    stats['iqr'] = stats['75%'] - stats['25%']
    stats['cluster'] = cluster_name
    return stats

# Compute statistics for each cluster
stats_data0 = compute_cluster_statistics(data0, 'Cluster 0')
stats_data1 = compute_cluster_statistics(data1, 'Cluster 1')
stats_data2 = compute_cluster_statistics(data2, 'Cluster 2')

# Combine the statistics into a single DataFrame
combined_stats = pd.concat([stats_data0, stats_data1, stats_data2])

# Reorder columns for better readability
combined_stats = combined_stats[['cluster', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 'iqr']]

# Display the combined statistics
print(combined_stats)


# # 3.1.1 Distributions Across Clusters

# In[32]:


# Plotting Visualisations

# Distribution of Player Overall Ratings by Cluster
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='overall', hue='cluster_optimal', kde=True, bins=30)
plt.title('Distribution of Player Overall Ratings by Cluster')
plt.xlabel('Overall Rating')
plt.ylabel('Count')
plt.show()


# In[33]:


# Distribution of Player Potential Ratings by Cluster
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='potential', hue='cluster_optimal', kde=True, bins=30)
plt.title('Distribution of Player Potential Ratings by Cluster')
plt.xlabel('Potential Rating')
plt.ylabel('Count')
plt.show()


# In[34]:


# Distribution of Attacking Finishing Ability by Cluster
plt.figure(figsize=(14, 8))
ax = sns.boxplot(data=data, x='cluster_optimal', y='attackingfinishing', palette='viridis')
plt.title('Distribution of Attacking Finishing Ability by Cluster')
plt.xlabel('Clusters')
plt.ylabel('Attacking Finishing Rating')

# Adding mean annotations
means = data.groupby('cluster_optimal')['attackingfinishing'].mean()
for cluster, mean in means.items():
    ax.text(cluster, mean, f'{mean:.2f}', horizontalalignment='center', verticalalignment='center', color='black')

plt.show()


# In[35]:


# Distribution of Passing Ability by Cluster
plt.figure(figsize=(14, 8))
ax = sns.boxplot(data=data, x='cluster_optimal', y='skilllongpassing', palette='viridis')
plt.title('Distribution of Long Passing Ability by Cluster')
plt.xlabel('Clusters')
plt.ylabel('Long Passing Rating')

# Adding mean annotations
means = data.groupby('cluster_optimal')['skilllongpassing'].mean()
for cluster, mean in means.items():
    ax.text(cluster, mean, f'{mean:.2f}', horizontalalignment='center', verticalalignment='center', color='black')

plt.show()


# In[36]:


# Distribution of Speed Ability by Cluster
plt.figure(figsize=(14, 8))
ax = sns.boxplot(data=data, x='cluster_optimal', y='movementsprintspeed', palette='viridis')
plt.title('Distribution of Sprint Speed by Cluster')
plt.xlabel('Clusters')
plt.ylabel('Sprint Speed Rating')

# Adding mean annotations
means = data.groupby('cluster_optimal')['movementsprintspeed'].mean()
for cluster, mean in means.items():
    ax.text(cluster, mean, f'{mean:.2f}', horizontalalignment='center', verticalalignment='center', color='black')

plt.show()


# In[37]:


# Distribution of Defensive Ability by Cluster
plt.figure(figsize=(14, 8))
ax = sns.boxplot(data=data, x='cluster_optimal', y='defendingmarkingawareness', palette='viridis')
plt.title('Distribution of Defensive Marking Awarness by Cluster')
plt.xlabel('Clusters')
plt.ylabel('Defensive Marking Awareness Rating')

# Adding mean annotations
means = data.groupby('cluster_optimal')['defendingmarkingawareness'].mean()
for cluster, mean in means.items():
    ax.text(cluster, mean, f'{mean:.2f}', horizontalalignment='center', verticalalignment='center', color='black')

plt.show()


# In[38]:


# Distribution of Goalkeeping Reflexes Ability by Cluster
plt.figure(figsize=(14, 8))
ax = sns.boxplot(data=data, x='cluster_optimal', y='goalkeepingreflexes', palette='viridis')
plt.title('Distribution of Goalkeeping Reflexes by Cluster')
plt.xlabel('Clusters')
plt.ylabel('Goalkeeping Reflexes Rating')

# Adding mean annotations
means = data.groupby('cluster_optimal')['goalkeepingreflexes'].mean()
for cluster, mean in means.items():
    ax.text(cluster, mean, f'{mean:.2f}', horizontalalignment='center', verticalalignment='center', color='black')

plt.show()


# In[39]:


# Distribution of Weak Foot Ability by Cluster
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='weakfoot', hue='cluster_optimal')
plt.title('Distribution of Weak Foot Ability by Cluster')
plt.xlabel('Weak Foot Rating')
plt.ylabel('Count')
plt.show()


# # 4. PCA For Dimensionality Reduction

# In[40]:


# Dimensionality Reduction using PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_data)


# In[41]:


# Add PCA results to DataFrame
data['pca1'] = pca_result[:, 0]
data['pca2'] = pca_result[:, 1]
data['pca3'] = pca_result[:, 2]


# In[42]:


# Alternatively, Matplotlib 3D Scatter Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data['pca1'], data['pca2'], data['pca3'],
                     c=data['cluster_optimal'], cmap='viridis', marker='o')

# Add color bar
legend1 = ax.legend(*scatter.legend_elements(), title='Cluster')
ax.add_artist(legend1)

# Labels and title
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('3D Scatter Plot of Clusters')

plt.show()


# # 5. Model Evaluation

# In[43]:


# Evaluation Metrics Computation (Using subsampling)
# Subsampling to reduce computation time
sample_size = 10000
indices = np.random.choice(scaled_data.shape[0], sample_size, replace=False)
sampled_scaled_data = scaled_data[indices]
sampled_labels = data['cluster_optimal'].iloc[indices]

# Calculate evaluation metrics on the sample
silhouette_avg_sampled = silhouette_score(sampled_scaled_data, sampled_labels)
davies_bouldin_avg_sampled = davies_bouldin_score(sampled_scaled_data, sampled_labels)
calinski_harabasz_avg_sampled = calinski_harabasz_score(sampled_scaled_data, sampled_labels)

# Print evaluation metrics for the sampled data
print(f"Silhouette Score (Sampled): {silhouette_avg_sampled:.4f}")
print(f"Davies-Bouldin Index (Sampled): {davies_bouldin_avg_sampled:.4f}")
print(f"Calinski-Harabasz Index (Sampled): {calinski_harabasz_avg_sampled:.4f}")


# # 6. Finding Similar (Closest) Players based on t-SNE Coordinates

# In[44]:


# Reduce dimensions with PCA before t-SNE
pca = PCA(n_components=50)
pca_result = pca.fit_transform(scaled_data)

# Subsample the data for t-SNE
sample_size = 10000 
sampled_pca_result = pca_result[:sample_size]

# Applying t-SNE to reduce dimensions to 2 for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(sampled_pca_result)

# Optionally, store the results in a DataFrame for easier plotting
tsne_df = pd.DataFrame(tsne_result, columns=['tsne1', 'tsne2'])


# In[45]:


# Plot t-SNE results
plt.figure(figsize=(12, 8))
sns.scatterplot(data=tsne_df, x='tsne1', y='tsne2', hue=data['cluster_optimal'].iloc[:sample_size], palette='tab10', s=50, alpha=0.7)
plt.title('t-SNE Visualization of Clusters')
plt.legend(title='Cluster')
plt.show()


# In[46]:


# Function to find the closest players
def find_closest_players(tsne_data, original_data, sample_size):
    df_results = pd.DataFrame()
    
    clusters = original_data['cluster_optimal'].iloc[:sample_size].unique()
    
    for cluster in clusters:
        # Select rows from the original data and tsne_data corresponding to the cluster
        cluster_indices = original_data[original_data['cluster_optimal'] == cluster].index
        cluster_indices = cluster_indices.intersection(tsne_data.index)
        df_cluster = original_data.loc[cluster_indices]
        coords = tsne_data.loc[cluster_indices][['tsne1', 'tsne2']].values
        names = df_cluster['shortname'].tolist()
        
        distances = cdist(coords, coords)
        np.fill_diagonal(distances, np.inf)  # Ensure diagonal isn't the closest
        indices = np.argpartition(distances, 10)[:, :10]
        closest_names = np.take(names, indices)
        
        closest_names_df = pd.DataFrame(closest_names, columns=[f'closest_{i}' for i in range(10)])
        closest_names_df['index_tracker'] = df_cluster.index
        
        df_results = pd.concat([df_results, closest_names_df])
    
    return df_results


# In[48]:


# Applying the function to find the closest players
closest_players_df = find_closest_players(tsne_df, data, sample_size)

# Merge the closest players DataFrame with the original DataFrame
data_with_closest = data.iloc[:sample_size].merge(closest_players_df, left_index=True, right_on='index_tracker')

# Save the DataFrame with the closest players to a CSV file
closest_players_filename = 'similar_players.csv'
data_with_closest.to_csv(closest_players_filename, index=False)


# In[49]:


# Load Similar players data

# Load the data
similar_players = pd.read_csv('similar_players.csv', low_memory=False)


# In[50]:


similar_players.head()


# In[ ]:




