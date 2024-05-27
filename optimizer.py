from typing import Callable

import numpy as np


def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))


def cost_matrix_lines(points: np.ndarray, cost_function: Callable = distance) -> np.ndarray:
    n = len(points)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i % 2 == 0 and j == i + 1:
                cost_matrix[i, j] = 0.01
                cost_matrix[j, i] = 0.01
            else:
                cost_matrix[i, j] = cost_function(points[i], points[j])
                cost_matrix[j, i] = cost_function(points[j], points[i])
    return cost_matrix


def cost_matrix_lines1(points: np.ndarray, cost_function: Callable = distance) -> np.ndarray:
    n = len(points)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i % 2 == 0 and j == i + 1:
                cost_matrix[i, j] = 0.01
                cost_matrix[j, i] = 0.01
            else:
                cost_matrix[i, j] = cost_function(points[i], points[j])
                cost_matrix[j, i] = cost_function(points[j], points[i])
    return cost_matrix


def cost_matrix_with_start(points: np.ndarray, start_point: np.ndarray,
                           cost_function: Callable = distance) -> np.ndarray:
    n = len(points)
    cost_matrix = np.zeros((n + 1, n + 1))
    all_points = [start_point, *points]
    start_costs = [cost_function(start_point, point) for point in all_points]
    cost_matrix[0] = start_costs
    cost_matrix[:, 0] = start_costs
    cost_matrix[1:, 1:] = cost_matrix_lines(points, cost_function)
    return cost_matrix


def ant_colony(cost_matrix, start_point, n_ants=10, iter=60, alpha=1.4, beta=3.8, rho=0.1, seed=42,
               circle=True):
    np.random.seed(seed)
    n = len(cost_matrix)
    norm = np.max(cost_matrix)
    cost_matrix = cost_matrix / norm
    visibility_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if np.isclose(cost_matrix[i, j], 0):
                visibility_matrix[i, j] = 0
            else:
                visibility_matrix[i, j] = 1 / cost_matrix[i, j]
    pheromone_map = np.ones((n, n))
    routes = np.zeros((n_ants, n), dtype=int)
    routes[:, 0] = start_point
    best_route_cost = np.inf
    for i in range(iter):
        routes_costs = np.zeros(n_ants)
        for ant in range(n_ants):
            ant_visibility_matrix = visibility_matrix.copy()
            for j in range(n - 1):
                cur_loc = routes[ant, j]
                ant_visibility_matrix[:, cur_loc] = 0
                tau = np.power(pheromone_map[cur_loc], alpha)[:, None]
                teta = np.power(ant_visibility_matrix[cur_loc], beta)[:, None]
                probs = np.multiply(tau, teta)
                probs = probs / np.sum(probs)
                cum_probs = np.cumsum(probs)
                r = np.random.random()
                next_loc = np.nonzero(cum_probs > r)[0][0]
                routes[ant, j + 1] = next_loc
                routes_costs[ant] += cost_matrix[cur_loc, next_loc]
            if circle:
                routes_costs[ant] += cost_matrix[routes[ant, -1], 0]
        pheromone_map = (1 - rho) * pheromone_map
        candidate = np.argmin(routes_costs)
        candidate_route_cost = routes_costs[candidate]
        if candidate_route_cost < best_route_cost:
            best_route_cost = candidate_route_cost
            best_route = routes[candidate].copy()
        delta_pheromone = 1 / best_route_cost
        for i, j, in zip(best_route[:-1], best_route[1:]):
            pheromone_map[i, j] += delta_pheromone
    return best_route, best_route_cost * norm


def nearest_neighbor(cost_matrix, start_point_number, circle):
    n = len(cost_matrix)
    norm = np.max(cost_matrix)
    cost_matrix = cost_matrix / norm
    path = np.zeros(n, dtype=int)
    path[0] = start_point_number
    path_cost_matrix = cost_matrix.copy()
    path_cost = 0
    for i in range(n - 1):
        cur_loc = path[i]
        path_cost_matrix[:, cur_loc] = 0
        cur_moves = path_cost_matrix[cur_loc]
        next_loc = np.where(cur_moves == np.min(cur_moves[np.nonzero(cur_moves)]))[0][0]
        path[i + 1] = next_loc
        path_cost += path_cost_matrix[cur_loc, next_loc]
    if circle:
        path_cost += cost_matrix[path[-1], path[0]]
    return path, path_cost * norm


def repetitive_nearest_neighbor(cost_matrix, circle):
    n = len(cost_matrix)
    paths = np.zeros((n, n), dtype=int)
    paths_cost = np.zeros(n)
    for i in range(n):
        paths[i], paths_cost[i] = nearest_neighbor(cost_matrix, i, circle)
    best_path = np.argmin(paths_cost)
    return paths[best_path], paths_cost[best_path]


def path_cost(path, cost_matrix, circle):
    if circle:
        cost = np.sum(cost_matrix[np.roll(path, 1), path])
    else:
        cost = np.sum(cost_matrix[np.roll(path, 1)[1:], path[1:]])
    return cost


def check_path(path):
    for i in range(1, len(path), 2):
        if path[i - 1] != path[i] - 1 and path[i - 1] != path[i] + 1:
            return False
    return True


def two_opt(path, cost_matrix, circle):
    best = path
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path)):
                new_route = path.copy()
                new_route[i:j] = path[j - 1:i - 1:-1]
                if check_path(new_route) and path_cost(new_route, cost_matrix, circle) < path_cost(
                        best, cost_matrix, circle):
                    best = new_route
                    improved = True
        path = best
    return best


def roll_path(path):
    start_point_ind = np.where(path == 0)[0][0]
    return np.roll(path, len(path) - start_point_ind)