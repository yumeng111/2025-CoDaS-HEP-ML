import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


def load_house_data(filename):
    """Load housing data with apartment/house labels"""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    sizes = data[:, 0]
    prices = data[:, 1] 
    labels = data[:, 2].astype(int)  # 0 = apartment, 1 = house
    return sizes, prices, labels


def plot_housing_prices(sizes, prices, scaled=False):
    """Plot house prices."""
    plt.figure(figsize=(6, 4))
    plt.scatter(sizes, prices, label='Data points', color='blue')
    plt.title('House Prices vs Size')
    if scaled:
        plt.xlabel('Size [a.u.]')
        plt.ylabel('Price [a.u.]')
    else:
        plt.xlabel('Size [sqm]')
        plt.ylabel('Price [CHF]')
    plt.legend()
    plt.grid()
    # plt.show()


def plot_housing_data_classified(sizes, prices, labels, scaled=False):
    """Plot the housing data colored by apartment/house classification"""
    apartments = labels == 0
    houses = labels == 1
    
    plt.figure(figsize=(4, 3))
    plt.scatter(sizes[apartments], prices[apartments]/1000, 
               label='Appartements', color='blue', alpha=0.7, s=50)
    plt.scatter(sizes[houses], prices[houses]/1000, 
               label='Houses', color='red', alpha=0.7, s=50)
    if scaled:
        plt.xlabel('Size [a.u.]')
        plt.ylabel('Price [a.u.]')
    else:
        plt.xlabel('Size (sqm)')
        plt.ylabel('Price [1k CHF]')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss_landscape(W, B, Loss):
    """Plot the loss landscape in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(W, B, Loss, cmap='viridis', alpha=0.7, edgecolor='none')
    
    ax.set_xlabel('Weight (w)')
    ax.set_ylabel('Bias (b)')
    ax.set_zlabel('Loss (MSE)')
    ax.set_title('Loss Landscape')
    
    # Add color bar
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    
    plt.show()


def plot_fit_landscape_and_loss(W, B, Loss, x, y, w_path, b_path, loss_path):
    # Create the plots with three subplots
    fig = plt.figure(figsize=(20, 6))   

    # Plot 1: Original data and fitted line
    ax1 = fig.add_subplot(131)
    ax1.scatter(x, y, alpha=0.6, label='Scaled data', color='blue')
    # Plot the final fitted line
    final_w, final_b = w_path[-1], b_path[-1]
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = final_w * x_line + final_b
    ax1.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fitted line (w={final_w:.3f}, b={final_b:.3f})')
    ax1.set_xlabel('Scaled House Size')
    ax1.set_ylabel('Scaled House Price')
    ax1.set_title('Linear Regression on Scaled Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: 3D Loss Surface with gradient descent path
    ax2 = fig.add_subplot(132, projection='3d')
    surface = ax2.plot_surface(W, B, Loss, cmap='viridis', alpha=0.7, edgecolor='none')

    # Plot gradient descent path as connected points
    ax2.plot(w_path, b_path, loss_path, 'ro-', markersize=4, linewidth=2, alpha=0.8, label='Gradient descent path')

    # Add arrows showing gradient descent steps
    for i in range(len(w_path)-1):
        ax2.quiver(w_path[i], b_path[i], loss_path[i],
                w_path[i+1] - w_path[i],
                b_path[i+1] - b_path[i],
                loss_path[i+1] - loss_path[i],
                color='red', arrow_length_ratio=0.1, alpha=0.7)

    # Mark important points
    ax2.scatter([w_path[0]], [b_path[0]], [loss_path[0]],
            color='orange', s=100, marker='s', label='Start point')
    ax2.scatter([w_path[-1]], [b_path[-1]], [loss_path[-1]],
            color='green', s=100, marker='o', label='End point')

    ax2.set_xlabel('Weight (w)')
    ax2.set_ylabel('Bias (b)')
    ax2.set_zlabel('Loss (MSE)')
    ax2.set_title('3D Loss Surface with Gradient Descent')
    ax2.legend()
    # change angle for better view
    ax2.view_init(elev=17, azim=-70)

    # Plot 3: Loss value as a function of steps
    ax3 = fig.add_subplot(133)
    steps = np.arange(len(loss_path))
    ax3.plot(steps, loss_path, 'b-o', linewidth=2, markersize=4, label='Loss')
    ax3.set_xlabel('Iteration Step')
    ax3.set_ylabel('Loss (MSE)')
    ax3.set_title('Loss Convergence During Training')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Add text showing final loss
    ax3.text(0.7, 0.9, f'Final Loss: {loss_path[-1]:.4f}', 
            transform=ax3.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()


# Alternative version with more control over animation speed
def animate_gradient_descent(
    W, B, Loss, x, y, w_path, b_path, loss_path, save_path=None
):
    """
    Version with custom speed - slow for first few steps, then faster
    """
    
    fig = plt.figure(figsize=(20, 6))
    
    # Same setup as above...
    ax1 = fig.add_subplot(131)
    ax1.scatter(x, y, alpha=0.6, color='blue', label='Scaled data')
    ax1.set_xlabel('Scaled House Size')
    ax1.set_ylabel('Scaled House Price')
    ax1.set_title('Linear Regression on Scaled Data')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    x_line = np.linspace(x.min(), x.max(), 100)
    line1, = ax1.plot([], [], 'r-', linewidth=2)
    
    ax2 = fig.add_subplot(132, projection='3d')
    surface = ax2.plot_surface(W, B, Loss, cmap='viridis', alpha=0.7, edgecolor='none')
    ax2.set_xlabel('Weight (w)')
    ax2.set_ylabel('Bias (b)')
    ax2.set_zlabel('Loss (MSE)')
    ax2.set_title('3D Loss Surface with Gradient Descent')
    ax2.view_init(elev=17, azim=-70)
    
    path_line, = ax2.plot([], [], [], 'ro-', markersize=4, linewidth=2, alpha=0.8)
    start_point = ax2.scatter([], [], [], color='orange', s=100, marker='s')
    current_point = ax2.scatter([], [], [], color='red', s=100, marker='o')
    
    ax3 = fig.add_subplot(133)
    ax3.set_xlabel('Iteration Step')
    ax3.set_ylabel('Loss (MSE)')
    ax3.set_title('Loss Convergence During Training')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, len(loss_path)-1)
    ax3.set_ylim(min(loss_path) * 0.95, max(loss_path) * 1.05)
    
    loss_line, = ax3.plot([], [], 'b-o', linewidth=2, markersize=4)
    
    text_ax1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    text_ax3 = ax3.text(0.02, 0.98, '', transform=ax3.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    def animate(frame):
        step = min(frame, len(w_path) - 1)
        
        # Update fitted line
        current_w = w_path[step]
        current_b = b_path[step]
        y_line = current_w * x_line + current_b
        line1.set_data(x_line, y_line)
        
        text_ax1.set_text(f'Step: {step}\nw: {current_w:.4f}\nb: {current_b:.4f}')
        
        # Update path
        if step > 0:
            path_line.set_data_3d(w_path[:step+1], b_path[:step+1], loss_path[:step+1])
            current_point._offsets3d = ([current_w], [current_b], [loss_path[step]])
        
        if step == 0:
            start_point._offsets3d = ([w_path[0]], [b_path[0]], [loss_path[0]])
        
        # Update loss plot
        steps_so_far = np.arange(step + 1)
        loss_so_far = loss_path[:step + 1]
        loss_line.set_data(steps_so_far, loss_so_far)
        
        text_ax3.set_text(f'Step: {step}\nCurrent Loss: {loss_path[step]:.6f}')
        
        return line1, path_line, start_point, current_point, loss_line, text_ax1, text_ax3
      
    n_frames = len(w_path) + 5
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=400,
                        blit=False, repeat=True)

    return anim
    
    # if save_path is None:
    #     plt.show()
    # else:
    #     anim.save("output/gradient-descent-animation.mp4", writer='ffmpeg')


def animate_gradient_descent_multivariate(
    features, x, y, params_path, loss_path, save_path=None
):
    """
    Version with custom speed - slow for first few steps, then faster
    """
    print(f"{x.shape=}, {y.shape=}, {len(params_path)=}, {len(loss_path)=}")
    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as patches
    
    fig = plt.figure(figsize=(20, 6))
    
    # Same setup as above...
    ax1 = fig.add_subplot(121)
    ax1.scatter(x, y, alpha=0.6, color='blue', label='Scaled data')
    ax1.set_xlabel('Scaled House Size')
    ax1.set_ylabel('Scaled House Price')
    ax1.set_title('Linear Regression on Scaled Data')
    ax1.set_xlim(x.min() *2, x.max() * 2)
    ax1.set_ylim(y.min() * 2, y.max() * 2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    x_line = np.linspace(x.min()*2, x.max()*2, 200)
    line1, = ax1.plot([], [], 'r-', linewidth=2)
        
    ax3 = fig.add_subplot(122)
    ax3.set_xlabel('Iteration Step')
    ax3.set_ylabel('Loss (MSE)')
    ax3.set_title('Loss Convergence During Training')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, len(loss_path)-1)
    ax3.set_ylim(min(loss_path) * 0.95, max(loss_path) * 1.05)
    
    loss_line, = ax3.plot([], [], 'b-o', linewidth=2, markersize=4)
    
    text_ax1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    text_ax3 = ax3.text(0.02, 0.98, '', transform=ax3.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    def animate(frame):
        step = min(frame, len(loss_path) - 1)
        # Update fitted line
        y_line = sum(params_path[step][i] * features[i](x_line) for i in range(len(features)))
        line1.set_data(x_line, y_line)
        
        # Update loss plot
        steps_so_far = np.arange(step + 1)
        loss_so_far = loss_path[:step + 1]
        loss_line.set_data(steps_so_far, loss_so_far)
        
        text_ax3.set_text(f'Step: {step}\nCurrent Loss: {loss_path[step]:.6f}')
        
        # return line1, path_line, start_point, current_point, loss_line, text_ax1, text_ax3
      
    n_frames = len(loss_path) + 5
    print(f"Number of frames: {n_frames}")
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=400,
                         blit=False, repeat=True)

    # return anim
    
    if save_path is None:
        plt.show()
    # else:
    #     anim.save("output/gradient-descent-animation.mp4", writer='ffmpeg')


def create_decision_boundary_mesh(weights, bias, x1_range, x2_range, grid_size=100):
    """Create decision boundary mesh for visualization (using first 2 features for plotting)"""
    x1_grid = np.linspace(x1_range[0], x1_range[1], grid_size)
    x2_grid = np.linspace(x2_range[0], x2_range[1], grid_size)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
    
    # Create feature mesh for all features
    feature_mesh = np.zeros((grid_size, grid_size, len(weights)))
    feature_mesh[:, :, 0] = X1_mesh        # size
    feature_mesh[:, :, 1] = X2_mesh        # price
    feature_mesh[:, :, 2] = X1_mesh ** 2   # size^2
    feature_mesh[:, :, 3] = X2_mesh ** 2   # price^2
    
    # Add more features here if needed...
    # feature_mesh[:, :, 4] = X1_mesh * X2_mesh  # size * price interaction
    
    # Calculate decision function values
    Z = np.zeros_like(X1_mesh)
    for i in range(len(weights)):
        Z += weights[i] * feature_mesh[:, :, i]
    Z += bias
    
    return X1_mesh, X2_mesh, Z


def animate_logistic_regression(features, labels, weights_history, bias_history, loss_history, 
                                feature_names, save_path=None):
    """Create animation of logistic regression training"""
    apartments = labels == 0
    houses = labels == 1
    
    # Use first two features for plotting (size and price)
    x1_data = features[:, 0]  # size
    x2_data = features[:, 1]  # price
    
    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define plot ranges
    x1_margin = (x1_data.max() - x1_data.min()) * 0.1
    x2_margin = (x2_data.max() - x2_data.min()) * 0.1
    x1_range = [x1_data.min() - x1_margin, x1_data.max() + x1_margin]
    x2_range = [x2_data.min() - x2_margin, x2_data.max() + x2_margin]
    
    # Plot 1: Decision boundary in feature space
    ax1.scatter(x1_data[apartments], x2_data[apartments], 
               color='blue', alpha=0.7, s=60, label='Apartments', edgecolor='darkblue')
    ax1.scatter(x1_data[houses], x2_data[houses], 
               color='red', alpha=0.7, s=60, label='Houses', edgecolor='darkred')
    
    # Store contour collections for cleanup
    contour_collections = []
    
    ax1.set_xlim(x1_range[0], x1_range[1])
    ax1.set_ylim(x2_range[0], x2_range[1])
    ax1.set_xlabel(f'{feature_names[0]} (scaled)', fontsize=12)
    ax1.set_ylabel(f'{feature_names[1]} (scaled)', fontsize=12)
    ax1.set_title('Non-linear Decision Boundary Evolution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss history
    loss_line, = ax2.plot([], [], 'blue', linewidth=2)
    ax2.set_xlim(0, len(loss_history))
    ax2.set_ylim(min(loss_history) * 0.95, max(loss_history) * 1.05)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Animation function
    def animate(frame):
        # Clear previous contour - more robust approach
        for contour_set in contour_collections:
            try:
                # Try the standard way first
                for collection in contour_set.collections:
                    collection.remove()
            except AttributeError:
                # Fallback for different matplotlib versions
                try:
                    contour_set.remove()
                except (AttributeError, ValueError):
                    # If all else fails, just skip removal
                    pass
        
        contour_collections.clear()
        
        # Update decision boundary
        current_weights = weights_history[frame]
        current_bias = bias_history[frame]
        X1_mesh, X2_mesh, Z = create_decision_boundary_mesh(current_weights, current_bias, x1_range, x2_range)
        
        # Draw decision boundary (where Z = 0)
        contour = ax1.contour(X1_mesh, X2_mesh, Z, levels=[0], colors=['green'], linewidths=3)
        contour_collections.append(contour)
        
        # Update loss plot
        loss_line.set_data(range(frame + 1), loss_history[:frame + 1])
        
        # Update titles
        ax1.set_title(f'Non-linear Decision Boundary (Iteration {frame})', fontsize=14)
        ax2.set_title(f'Training Loss (Current: {loss_history[frame]:.4f})', fontsize=14)
        
        return loss_line,
    
    # Create animation
    frames = len(weights_history)
    # frames = min(len(weights_history), 200)
    interval = max(100, len(weights_history) // frames * 50)
    
    anim = FuncAnimation(fig, animate, frames=frames, interval=interval, 
                                 blit=False, repeat=True)
    
    plt.tight_layout()
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=10, bitrate=1800)
        print("Animation saved!")
    
    plt.show()
    return anim


def create_decision_boundary_mesh_nn(func, x1_range, x2_range):
    """
    Create a mesh grid and compute neural network predictions for decision boundary
    
    Parameters:
    - nn: neural network with current weights
    - x1_range: [min, max] for first feature (size)  
    - x2_range: [min, max] for second feature (price)
    
    Returns:
    - X1_mesh, X2_mesh: meshgrid coordinates
    - Z: neural network predictions (probabilities - 0.5 for boundary)
    """
    # Create mesh grid
    x1 = np.linspace(x1_range[0], x1_range[1], 100)
    x2 = np.linspace(x2_range[0], x2_range[1], 100)
    X1_mesh, X2_mesh = np.meshgrid(x1, x2)
    
    # Create features for mesh points (same as training features)
    mesh_points = np.c_[X1_mesh.ravel(), X2_mesh.ravel()]
    x1_vals = mesh_points[:, 0]
    x2_vals = mesh_points[:, 1]
    
    # Create feature matrix matching training data format
    features_mesh = np.column_stack([
        x1_vals,           # Feature 0: size (scaled)
        x2_vals,           # Feature 1: price (scaled)  
    ])
    
    # Get predictions and reshape
    predictions = func(features_mesh)
    Z = (predictions - 0.5).reshape(X1_mesh.shape)  # Center around 0 for contour
    
    return X1_mesh, X2_mesh, Z


def animate_neural_network(
    features, lamdas, labels, loss_history, feature_names, 
    plot_every=1, save_path=None
):
    """Create animation of neural network training"""
    plot_every = int(plot_every)
    lamdas = lamdas[::plot_every]  # Downsample labels for animation
    loss_history = loss_history[::plot_every]  # Downsample loss history

    apartments = labels == 0
    houses = labels == 1
    
    # Use first two features for plotting (size and price - scaled)
    x1_data = features[:, 0]  # size (scaled)
    x2_data = features[:, 1]  # price (scaled)
    
    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define plot ranges
    x1_margin = (x1_data.max() - x1_data.min()) * 0.1
    x2_margin = (x2_data.max() - x2_data.min()) * 0.1
    x1_range = [x1_data.min() - x1_margin, x1_data.max() + x1_margin]
    x2_range = [x2_data.min() - x2_margin, x2_data.max() + x2_margin]
    
    # Plot 1: Decision boundary in feature space
    ax1.scatter(x1_data[apartments], x2_data[apartments],
                color='blue', alpha=0.7, s=60, label='Apartments', edgecolor='darkblue')
    ax1.scatter(x1_data[houses], x2_data[houses], 
                color='red', alpha=0.7, s=60, label='Houses', edgecolor='darkred')
    
    # Store contour collections for cleanup
    contour_collections = []
    
    ax1.set_xlim(x1_range[0], x1_range[1])
    ax1.set_ylim(x2_range[0], x2_range[1])
    ax1.set_xlabel(f'{feature_names[0]} (scaled)', fontsize=12)
    ax1.set_ylabel(f'{feature_names[1]} (scaled)', fontsize=12)
    ax1.set_title('Neural Network Decision Boundary Evolution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss history
    loss_line, = ax2.plot([], [], 'blue', linewidth=2)
    ax2.set_xlim(0, len(loss_history)*plot_every)
    ax2.set_ylim(min(loss_history) * 0.95, max(loss_history) * 1.05)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss', fontsize=14)
    ax2.grid(True, alpha=0.3)
        
    # Animation function
    def animate(frame):
        # Clear previous contour - more robust approach
        for contour_set in contour_collections:
            try:
                # Try the standard way first
                for collection in contour_set.collections:
                    collection.remove()
            except AttributeError:
                # Fallback for different matplotlib versions
                try:
                    contour_set.remove()
                except (AttributeError, ValueError):
                    # If all else fails, just skip removal
                    pass
        contour_collections.clear()
                
        # Update decision boundary
        X1_mesh, X2_mesh, Z = create_decision_boundary_mesh_nn(lamdas[frame], x1_range, x2_range)
        
        # Draw decision boundary (where Z = 0, i.e., probability = 0.5)
        contour = ax1.contour(X1_mesh, X2_mesh, Z, levels=[0], colors=['green'], linewidths=3)
        contour_collections.append(contour)
        
        # Update loss plot
        loss_line.set_data([e*plot_every for e in range(frame + 1)], loss_history[:frame + 1])
        
        # Update titles
        ax1.set_title(f'Neural Network Decision Boundary (Iteration {frame*plot_every})', fontsize=14)
        ax2.set_title(f'Training Loss (Current: {loss_history[frame]:.4f})', fontsize=14)
        
        return loss_line,
    
    # Create animation
    frames = len(loss_history)
    interval = max(100, len(loss_history) // frames * 50)
    
    anim = FuncAnimation(fig, animate, frames=frames, interval=interval,
                                 blit=False, repeat=True)
    
    plt.tight_layout()
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=10, bitrate=1800)
        print("Animation saved!")
    
    # plt.show()
    return anim