import numpy as np
import matplotlib.pyplot as plt


def create_spider_plot(data, nfl_id, component_names):
    # Extract the player's scores for the first 7 components
    pca_scores = data.set_index("nfl_id")
    player_name = nfl_id
    player_data = pca_scores.loc[nfl_id, component_names]

    # Get all values for scaling
    all_values = pca_scores[component_names].values.flatten()
    
    # Find the minimum value across all players and components
    min_value = np.min(all_values)
    max_value = np.max(all_values)
    
    # Rescale all values so minimum is 0
    scaled_values = player_data - min_value

    # Rescale all values to range from 0 to 1
    if max_value != min_value:  # Avoid division by zero
        scaled_values = (player_data - min_value) / (max_value - min_value)
    else:
        scaled_values = player_data - min_value  # All values are the same
    
    # Prepare data for plotting
    angles = np.linspace(0, 2 * np.pi, len(component_names), endpoint=False)
    
    # Close the loop
    values = np.concatenate((scaled_values, [scaled_values.iloc[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot the data
    ax.plot(angles, values, 'o-', linewidth=2, label=player_name)
    ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(component_names)
    ax.set_yticklabels([])
    
    # Set radial limits (add some padding)
    max_value = np.max(values)
    ax.set_ylim(0, 1.1)
        
    # Add title
    plt.title(f'PCA Components for {player_name}', size=16, pad=20)
    
    # Show the plot
    plt.show()