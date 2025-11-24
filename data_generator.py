import os
import re
from tsp_core import City

def load_tsp_file(path):
    """
    Universal TSPLIB loader.
    Supports:
        - EUC_2D
        - ATT
        - CEIL_2D
        - GEO
    Handles:
        - lowercase/uppercase section names
        - blank lines
        - comments
        - weird formatting (gr48.tsp, eil51, att48, etc.)
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"TSP file not found: {path}")

    with open(path, "r") as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    # Normalize lines for easier detection
    lines_upper = [l.upper() for l in raw_lines]

    # --------------------------------------------
    # 1. Find the start of NODE_COORD_SECTION
    # --------------------------------------------
    start_index = None
    for i, line in enumerate(lines_upper):
        if "NODE_COORD_SECTION" in line:
            start_index = i + 1
            break

    # Some TSPLIB files omit NODE_COORD_SECTION and start coordinates directly
    if start_index is None:
        # Try to detect coordinate lines directly
        for i, line in enumerate(raw_lines):
            if re.match(r"^\s*\d+\s+[-]?\d+(\.\d+)?\s+[-]?\d+(\.\d+)?", line):
                start_index = i
                break

    if start_index is None:
        raise ValueError(f"Could not find coordinate section in: {path}")

    # --------------------------------------------
    # 2. Parse coordinates
    # --------------------------------------------
    cities = []
    for line in raw_lines[start_index:]:
        if line.upper().startswith("EOF"):
            break

        # Ignore invalid lines
        if not re.match(r"^\d+", line):
            continue

        parts = re.split(r"\s+", line)
        if len(parts) < 3:
            continue

        try:
            # First column is index; skip it
            x = float(parts[1])
            y = float(parts[2])
            cities.append(City(x, y))
        except:
            continue

    if len(cities) == 0:
        raise ValueError(f"No coordinates parsed in: {path}")

    return cities
