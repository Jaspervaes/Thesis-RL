import numpy as np
import pickle

def save_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_data(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def generate_dash_patterns(num_patterns, max_segments=2, max_length=10):
    dash_patterns = []
    for _ in range(num_patterns):
        num_segments = np.random.randint(1, max_segments + 1)
        dash_pattern = np.random.randint(1, max_length + 1, size=num_segments * 2)
        dash_patterns.append(tuple(dash_pattern))
    return dash_patterns