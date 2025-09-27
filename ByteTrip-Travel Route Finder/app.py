from flask import Flask, render_template, request
import pandas as pd
import heapq
from collections import deque, defaultdict

app = Flask(__name__)

# ------- Data load -------
df = pd.read_csv("route_infos.csv")

# undirected adjacency list:
# graph[u] = list of (v, cost, distance_km, time_minutes)
graph = defaultdict(list)
for _, row in df.iterrows():
    u, v = row['from'], row['to']
    c, d, t = row['cost'], row['distance_km'], row['time_minutes']
    graph[u].append((v, c, d, t))
    graph[v].append((u, c, d, t))  # undirected


# ------- Algorithms -------
def dijkstra(graph, start, end, weight_index):
    """
    weight_index: 1=cost, 2=distance_km, 3=time_minutes
    """
    heap = [(0, start, [])]
    visited = set()
    while heap:
        cost_so_far, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        path = path + [node]
        visited.add(node)
        if node == end:
            return cost_so_far, path
        for (nbr, c, d, t) in graph[node]:
            if nbr not in visited:
                w = (c, d, t)[weight_index - 1]
                heapq.heappush(heap, (cost_so_far + w, nbr, path))
    return float('inf'), []


def bfs_fewest_stops(graph, start, end):
    """
    BFS finds the path with the FEWEST EDGES (fewest stops).
    Returns: (hops, path_list). hops = number of edges.
    """
    if start == end:
        return 0, [start]
    q = deque([start])
    parent = {start: None}
    while q:
        u = q.popleft()
        for (v, *_rest) in graph[u]:
            if v not in parent:
                parent[v] = u
                if v == end:
                    # reconstruct path
                    path = []
                    cur = v
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return len(path) - 1, path
                q.append(v)
    return float('inf'), []


def totals_along_path(graph, path):
    """
    Given a concrete path, sum (cost, distance, time) along it.
    If any edge missing, returns (inf, inf, inf).
    """
    if not path or len(path) == 1:
        return (0, 0, 0)
    total_cost = total_dist = total_time = 0
    for u, v in zip(path, path[1:]):
        found = False
        for (nbr, c, d, t) in graph[u]:
            if nbr == v:
                total_cost += c
                total_dist += d
                total_time += t
                found = True
                break
        if not found:
            return (float('inf'), float('inf'), float('inf'))
    return (total_cost, total_dist, total_time)


# ------- Helpers -------
def safe_num(v):
    """Turn float('inf') into None so templates stay simple."""
    return None if v == float('inf') else v


# ------- Routes -------
@app.route("/", methods=["GET", "POST"])
def index():
    districts = sorted(set(df['from']).union(df['to']))

    if request.method == "POST":
        start = request.form["start"]
        end = request.form["end"]

        # Same-source-destination early return
        if start == end:
            results = {
                "dijkstra": {
                    "cheapest": {"value": 0, "path": [start]},
                    "shortest_distance": {"value": 0, "path": [start]},
                    "shortest_time": {"value": 0, "path": [start]},
                },
                "bfs": {
                    "fewest_stops": {"hops": 0, "path": [start]},
                    "totals": {"cost": 0, "distance": 0, "time": 0}
                },
                "complexity": {
                    "dijkstra": "O((V + E) log V) with a binary heap",
                    "bfs": "O(V + E)"
                },
                "note": "Start and destination are the same. Everything is zero—budget-friendly AND punctual!"
            }
            return render_template("result.html", results=results)

        # Dijkstra: true optima for each weighted metric
        cheapest_cost, path_cheapest = dijkstra(graph, start, end, weight_index=1)
        shortest_distance, path_shortest_dist = dijkstra(graph, start, end, weight_index=2)
        shortest_time, path_shortest_time = dijkstra(graph, start, end, weight_index=3)

        # BFS: fewest stops; then sum weights along that path for comparison
        hops, path_fewest_stops = bfs_fewest_stops(graph, start, end)
        bfs_cost, bfs_dist, bfs_time = totals_along_path(graph, path_fewest_stops)

        results = {
            "dijkstra": {
                "cheapest": {"value": safe_num(cheapest_cost), "path": path_cheapest},
                "shortest_distance": {"value": safe_num(shortest_distance), "path": path_shortest_dist},
                "shortest_time": {"value": safe_num(shortest_time), "path": path_shortest_time},
            },
            "bfs": {
                "fewest_stops": {"hops": safe_num(hops), "path": path_fewest_stops},
                "totals": {
                    "cost": safe_num(bfs_cost),
                    "distance": safe_num(bfs_dist),
                    "time": safe_num(bfs_time),
                }
            },
            "complexity": {
                "dijkstra": "O((V + E) log V) with a binary heap",
                "bfs": "O(V + E)"
            },
            "note": (
                "BFS gives the FEWEST-STOPS path. Its totals (cost/distance/time) are along that path "
                "and may NOT be minimal for those metrics. Dijkstra’s results are the true minima."
            )
        }

        # Add a heads-up if anything was unreachable
        if any(x is None for x in [
            results["dijkstra"]["cheapest"]["value"],
            results["dijkstra"]["shortest_distance"]["value"],
            results["dijkstra"]["shortest_time"]["value"],
            results["bfs"]["fewest_stops"]["hops"],
        ]):
            results["note"] += " At least one route appears unreachable with the current data."

        return render_template("result.html", results=results)

    return render_template("index.html", districts=districts)


if __name__ == "__main__":
    app.run(debug=True)
