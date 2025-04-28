"""
Grid management module for map processing and path planning.
"""
import numpy as np
import pandas as pd
from Project3.config import CUBE_SIZE_METERS

def load_grid_from_csv(csv_path):
    """
    Reads a CSV file and converts it to a numpy array.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        A numpy array representation of the grid
    """
    df = pd.read_csv(csv_path, header=None)
    return np.array(df)

def pad_grid(grid, radius):
    """
    Adds padding around obstacles in the grid.
    
    Args:
        grid: The grid to pad
        radius: The radius around obstacles to pad
        
    Returns:
        A padded grid
    """
    padded_grid = np.copy(grid)
    obstacles = np.where(grid == 1)
    for i, j in zip(obstacles[0], obstacles[1]):
        # Calculate all cells in the radius
        for x in range(max(0, i - radius), min(len(grid), i + radius + 1)):
            for y in range(max(0, j - radius), min(len(grid[i]), j + radius + 1)):
                padded_grid[x][y] = 1
    
    return padded_grid

def upscale_grid(grid, upscaling_factor=4):
    """
    Upscales a grid by the given factor.
    
    Args:
        grid: The grid to upscale
        upscaling_factor: The factor by which to upscale the grid
        
    Returns:
        An upscaled grid
    """
    upscale_factor = upscaling_factor * 2 - 1  # ensure odd number
    
    upscaled_grid = np.zeros(
        (len(grid) * upscale_factor, len(grid[0]) * upscale_factor)
    )
    
    for x, row in enumerate(grid):
        for y, cell in enumerate(row):
            if cell in [2, 3, 4, 5]:  # Special markers
                upscaled_grid[x * upscale_factor + upscale_factor // 2, 
                              y * upscale_factor + upscale_factor // 2] = cell
            else:
                upscaled_grid[x * upscale_factor:(x + 1) * upscale_factor,
                              y * upscale_factor:(y + 1) * upscale_factor] = cell
    
    return upscaled_grid

def process_grid(grid, upscaling_factor=4):
    """
    Processes a grid by upscaling and padding it.
    
    Args:
        grid: The input grid
        upscaling_factor: The factor by which to upscale the grid
        
    Returns:
        A processed grid
    """
    upscale_factor = upscaling_factor * 2 - 1  # ensure odd number
    
    upscaled_grid = upscale_grid(grid, upscaling_factor)
    
    padded_grid = pad_grid(upscaled_grid, radius=max(upscale_factor - 1, 1))
    
    return padded_grid

def find_position_in_grid(grid, position_value):
    """
    Find coordinates of a specific value in the grid.
    
    Args:
        grid: The grid to search
        position_value: The value to find
        
    Returns:
        (x, y) coordinates in the grid, or (None, None) if not found
    """
    positions = np.where(grid == position_value)
    
    if len(positions[0]) > 0 and len(positions[1]) > 0:
        # Get the first matching position
        grid_x, grid_y = positions[0][0], positions[1][0]
        return grid_x, grid_y
    else:
        return None, None

def grid_to_world_coords(grid_x, grid_y):
    """
    Convert grid coordinates to world coordinates.
    
    Args:
        grid_x: X coordinate in grid
        grid_y: Y coordinate in grid
        
    Returns:
        (x, y) coordinates in world frame
    """
    if grid_x is None or grid_y is None:
        return None, None
        
    world_x = grid_x * CUBE_SIZE_METERS
    world_y = grid_y * CUBE_SIZE_METERS
    return world_x, world_y
