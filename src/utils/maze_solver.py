import heapq

import json5
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(node, grid):
    rows, cols = grid.shape
    for dr, dc in dirs:
        r, c = node[0] + dr, node[1] + dc
        if rows > r >= 0 == grid[r, c] and 0 <= c < cols:
            yield r, c


def astar_with_history(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    closed = set()
    history = []

    while open_set:
        f, g, current = heapq.heappop(open_set)

        history.append({
            'open': [node for (_, _, node) in open_set],
            'closed': list(closed),
            'current': current
        })

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path = path[::-1]

            history.append({
                'open': [],
                'closed': list(closed),
                'current': goal,
                'path': path
            })
            return path, history

        closed.add(current)
        for nbr in neighbors(current, grid):
            if nbr in closed:
                continue
            tentative = g_score[current] + 1
            if tentative < g_score.get(nbr, float('inf')):
                came_from[nbr] = current
                g_score[nbr] = tentative
                heapq.heappush(open_set, (tentative + heuristic(nbr, goal), tentative, nbr))

    return None, history


def visualize(grid, history, start, goal, interval=200):
    fig, ax = plt.subplots()

    cmap = plt.cm.Greens.copy()
    cmap.set_bad(color='black')

    initial = np.ma.masked_where(grid == 1, grid.astype(float))
    im = ax.imshow(initial, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([]);
    ax.set_yticks([])

    start_scatter = ax.scatter(start[1], start[0],
                               marker='s', color='yellow', s=100, label='Start')
    goal_scatter = ax.scatter(goal[1], goal[0],
                              marker='s', color='red', s=100, label='Goal')
    path_scatter = ax.scatter([], [], marker='s', color='blue', s=100, label='Path')

    def update(frame):
        data = history[frame]
        disp = grid.astype(float).copy()

        for (r, c) in data['closed']:
            disp[r, c] = 0.4
        for (r, c) in data['open']:
            disp[r, c] = 0.7
        cr, cc = data['current']
        disp[cr, cc] = 1.0

        disp = np.ma.masked_where(grid == 1, disp)
        im.set_array(disp)

        if 'path' in data:
            coords = [(c, r) for r, c in data['path']]
            path_scatter.set_offsets(coords)

        return [im, path_scatter, start_scatter, goal_scatter]

    ani = animation.FuncAnimation(
        fig, update, frames=len(history),
        interval=interval, blit=True, repeat=False
    )

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.show(block=True)
    return ani


def load_maze(file_path="../../scenarios/_maze_home_sweet_home.json5") -> np.ndarray:
    """
    Reads a JSON5 file containing an array and returns it as a NumPy array.

    Args:
        file_path: The path to the JSON5 file.

    Returns:
        A NumPy array containing the data from the file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json5.load(f)
        return np.array(data)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        raise
    except Exception as e:
        print(f"An error occurred while parsing '{file_path}': {e}")
        raise


def solve_maze():
    """
    Solve the maze using A* algorithm and visualize the path with history.
    """
    grid = load_maze()
    start = (38, 6)
    goal = (30, 52)

    path, history = astar_with_history(grid, start, goal)
    visualize(grid, history, start, goal)


if __name__ == "__main__":
    solve_maze()