import json
import re
import sys
from pathlib import Path

def convert_py_to_ipynb(input_py_path, output_ipynb_path=None):
    """
    Convert a Python script to a Jupyter Notebook with cells separated by #@title markers.
    
    Args:
        input_py_path (str): Path to the Python script
        output_ipynb_path (str, optional): Path for the output notebook. If None, uses the same name with .ipynb extension.
    
    Returns:
        str: Path to the created notebook file
    """
    # Read the Python file
    with open(input_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # If no output path is provided, use the same name with .ipynb extension
    if output_ipynb_path is None:
        output_ipynb_path = str(Path(input_py_path).with_suffix('.ipynb'))
    
    # Split the content into cells based on #@title markers
    # We want to include the #@title line in each cell
    cell_contents = []
    
    # Handle the first part before any #@title if it exists
    parts = re.split(r'(#@title.*?\n)', content, flags=re.DOTALL)
    
    if not parts[0].strip().startswith('#@title'):
        # If the file doesn't start with #@title, add the first part as a cell
        if parts[0].strip():  # Only add if there's actual content
            cell_contents.append(parts[0])
    
    # Process the rest of the parts (title markers and content)
    i = 0
    while i < len(parts):
        if parts[i].strip().startswith('#@title'):
            # This is a title marker
            title_line = parts[i]
            # Look for the next content part
            if i + 1 < len(parts):
                cell_contents.append(title_line + parts[i+1])
                i += 2
            else:
                # If this is the last part and it's a title marker, add it by itself
                cell_contents.append(title_line)
                i += 1
        else:
            # Skip any content part that we didn't handle with a title
            i += 1
    
    # Create the notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Process each cell
    for cell_content in cell_contents:
        # Extract the title if it exists
        title_match = re.match(r'#@title (.*?)(\n|$)', cell_content)
        title = title_match.group(1) if title_match else ""
        
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": cell_content.splitlines(True)  # Keep the newlines
        }
        
        # Add the title to cell metadata if it exists
        if title:
            cell["metadata"]["title"] = title
            # Set the cell to be collapsed by default
            cell["metadata"]["collapsed"] = True
            # Also add folding flag for further compatibility
            cell["metadata"]["cellView"] = "form"
        
        notebook["cells"].append(cell)
    
    # Write the notebook to file
    with open(output_ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    return output_ipynb_path

def main():
    """
    Main function to process command line arguments.
    
    Usage:
        python convert_to_ipynb.py input_file.py [output_file.ipynb]
    """
    if len(sys.argv) < 2:
        print("Usage: python convert_to_ipynb.py input_file.py [output_file.ipynb]")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_path = convert_py_to_ipynb(input_file, output_file)
        print(f"Conversion successful! Notebook saved to: {output_path}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    main()
