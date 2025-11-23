# Dijkstra's Algorithm Implementation

This project implements Dijkstra's algorithm for finding the shortest paths in weighted graphs. Dijkstra's algorithm is a graph search algorithm that solves the single-source shortest path problem for graphs with non-negative edge weights.

## Features

- Implementation of Dijkstra's algorithm for shortest path calculation
- Support for weighted graphs represented as adjacency lists
- Efficient implementation using priority queues
- Comprehensive testing and documentation
- Clean, modular code structure

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd dijkstra-algorithm

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from src.core.graph import Graph

# Create a graph
graph = Graph()

# Add edges to the graph
graph.add_edge('A', 'B', 4)
graph.add_edge('A', 'C', 2)
graph.add_edge('B', 'C', 1)
graph.add_edge('B', 'D', 5)
graph.add_edge('C', 'D', 8)
graph.add_edge('C', 'E', 10)
graph.add_edge('D', 'E', 2)

# Find shortest paths from source node 'A'
distances, previous = graph.dijkstra('A')

print("Shortest distances:", distances)
print("Previous nodes:", previous)
```

## Project Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── graph.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── docs/
│   └── index.md
└── config/
    └── settings.yaml
```

## Testing

Run tests with:

```bash
python -m pytest tests/
```

## License

MIT License