import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.manifold import MDS


def get_base64_scatter_plot(df, distance_matrix, specified_clusterA=None, specified_clusterB=None):
    # Ensure the cluster information is in string format for consistent color mapping
    df['cluster'] = df['cluster'].astype(str)

    # Calculate the size of each cluster
    cluster_sizes = df['cluster'].value_counts()

    # Prepare the colors for clusters
    cmap = plt.get_cmap('hsv')  # Using a colormap that supports more colors
    clusters = df['cluster'].unique()
    cluster_colors = {}

    legend_labels = {}
    if specified_clusterA is not None:
        # Color only the specified cluster in red, others are gray
        for cluster in clusters:
            if cluster == specified_clusterA:
                cluster_colors[cluster] = 'red'
                # legend_labels[cluster] = 'Specified Cluster'
            else:
                cluster_colors[cluster] = 'gray'

    if specified_clusterB is not None:
        # Color only the specified cluster in red, others are gray
        for cluster in clusters:
            if cluster == specified_clusterB:
                cluster_colors[cluster] = 'blue'
                # legend_labels[cluster] = 'Specified Cluster'
            else:
                cluster_colors[cluster] = 'gray'

    if specified_clusterA is None and specified_clusterB is None:
        # Color clusters with more than one member, single-member clusters are gray
        color_index = 0
        for cluster in clusters:
            if cluster_sizes[cluster] > 1:
                cluster_colors[cluster] = color_index
                color_index += 1
            else:
                cluster_colors[cluster] = 'gray'

        for cluster in clusters:
            if cluster_sizes[cluster] > 1:
                cluster_colors[cluster] = cmap(cluster_colors[cluster] / color_index)
                legend_labels[cluster] = f'Cluster {cluster}'

    # Extract students' IDs as a list
    students = df['student_id'].tolist()

    # Convert the distance matrix to a numpy array
    filtered_matrix = np.array([
        [distance_matrix.get(s1, {}).get(s2, 0) for s2 in students]
        for s1 in students
    ])

    # Apply Multidimensional Scaling (MDS) to reduce dimensions to 2D
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(filtered_matrix)

    # Plot the 2D scatter plot of students with colors representing their clusters
    plt.figure(figsize=(10, 8))
    for cluster, color in cluster_colors.items():
        # Select data points belonging to the current cluster
        cluster_coords = coords[df['cluster'] == cluster]
        plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], color=color, s=50, alpha=0.6, edgecolors='w',
                    label=legend_labels.get(cluster))

    # Create and adjust the legend
    if specified_clusterA is None and specified_clusterB is None:
        plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("2D Scatter Plot of Clustering")
    plt.tight_layout()
    return plt


def get_pie_chart(cluster_info):
    def get_label(key):
        if key == -1:
            return "Not Clustered"
        elif key == -2:
            return "Non Compliance"
        else:
            return str(key)  # Convert other keys to string without changing them

    labels = [get_label(key) for key in cluster_info.keys()]

    amounts = [amount[1] for amount in cluster_info.values()]

    # Create pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(amounts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Cluster Sizes')
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def get_progress_pie_chart(df):
    # Count entries where grade is -1 and entries where grade is not -1
    not_clustered_count = df[df['grade'] == -1].shape[0]
    clustered_count = df[df['grade'] != -1].shape[0]

    # Labels and amounts for the pie chart
    labels = ['Not Clustered', 'Clustered']
    amounts = [not_clustered_count, clustered_count]

    # Create pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(amounts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['gray', 'green'])
    plt.title('Distribution of Clustered vs Not Clustered Grades')
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()
