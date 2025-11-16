import re # Make sure 'import re' is at the top of the file

def load_tsp_file(file_path: str) -> list:
    """
    Loads a .tsp file (TSPLIB format) and returns a list of City objects.
    
    Args:
        file_path (str): The path to the .tsp file.

    Returns:
        list: A list of City objects.
    """
    cities = []
    # Regex to capture "node_id x_coord y_coord"
    # It handles integers and scientific notation (like in berlin52.tsp)
    coord_pattern = re.compile(
        r"^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$", 
        re.MULTILINE
    )
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Find the start of the coordinate section
            node_coord_section = re.search(r"NODE_COORD_SECTION", content)
            if not node_coord_section:
                print(f"Error: NODE_COORD_SECTION not found in {file_path}")
                return []
            
            # Find the end of the file or another section
            eof_section = re.search(r"EOF", content[node_coord_section.end():])
            
            if eof_section:
                coord_data = content[node_coord_section.end() : node_coord_section.end() + eof_section.start()]
            else:
                coord_data = content[node_coord_section.end():] # Read to end if EOF is missing
            
            # Find all matching coordinate lines
            matches = coord_pattern.findall(coord_data)
            
            if not matches:
                print(f"Error: No coordinates found in NODE_COORD_SECTION in {file_path}")
                return []

            for match in matches:
                # We don't use node_id (match[0]) but it's good for validation
                name = str(match[0])
                x = float(match[1])
                y = float(match[2])
                
                # We need to import the City class
                # Make sure this import is at the top of data_generator.py
                from tsp_core import City
                cities.append(City(x, y, name))
                
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []
        
    return cities