def find_paths(network, origin, target):
    stack = [(origin, [origin])]
    while stack:
        print("Stack:", stack)
        device, path = stack.pop()
        for neighbor in network[device]:
            if neighbor in path:
                continue
            next_path = path + [neighbor]
            if neighbor == target:
                print("Stack before return:", stack)
                return next_path
            stack.append((neighbor, next_path))

def get_trust_values(network, origin, target):
    all_paths = find_paths(network, origin, target)
    print("all_paths:", all_paths)
    trust_values = []
    for path in all_paths:
        if len(path) >= 2:  # Ensure path has at least two devices
            neighbors = network[path[-2]]  # Getting neighbors via path
            trust_values.extend(neighbors.get(path[-1], []))  # Retrieve trust scores if neighbor exists
    return trust_values

# Example usage:
network = {
    'A': {'B': 0.8, 'C': 0.6},
    'B': {'A': 0.9, 'D': 0.7},
    'C': {'A': 0.7, 'E': 0.5},
    'D': {'B': 0.6, 'F': 0.4},
    'E': {'C': 0.8, 'G': 0.6},
    'F': {'D': 0.9, 'G': 0.7},
    'G': {'E': 0.7, 'F': 0.5}
}

origin = 'A'
target = 'G'

trust_values = get_trust_values(network, origin, target)
print("Trust values/recommendations from neighbors:", trust_values)

