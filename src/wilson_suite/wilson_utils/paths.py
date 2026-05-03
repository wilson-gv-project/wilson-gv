import os
from pathlib import Path

UTILS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_ROOT = os.path.abspath(os.path.join(UTILS_ROOT, '..'))


WORKFLOW_BASE_DIR = Path(os.environ.get("WORKFLOW_BASE_DIR", "../workflows")).expanduser().resolve()
# .resolve() -- converts the path to an absolute path; normalizes . and ..

def update_filename(old_name, new_filename: str):
    """
    updating filename attribute
    """
    current_path = Path(old_name)
    new_path = Path(new_filename)

    if not new_path.is_absolute():

        if current_path.is_absolute():
            return str(current_path.parent / new_filename)
        else:
            return new_filename
    else:
        return new_filename


def make_filename_from(*, new_filename: str, template: str, keep: str, tag: str = "") -> Path:
    """
    Constructs a new path based on specific logic.
    
    modes: 
    - 'dir': Folder from template + Full name from new_filename.
    - 'dir+name': Folder from template + Stem from template + Ext from new_filename.
    - 'stem': Folder from new_filename + Stem from template + Ext from new_filename.
    """
    t = Path(template)
    n = Path(new_filename)

    if keep == "dir":
        base_path = t.parent / n.name
        
    elif keep == "dir+name":
        base_path = t.with_suffix(n.suffix)
        
    elif keep == "stem":
        # Update dir and extension, but KEEP template stem
        # Result: n.parent / t.stem + n.suffix
        base_path = n.with_name(f"{t.stem}{n.suffix}")
        
    else:
        raise ValueError(f"Unknown mode: {keep}")

    # Apply tag if present
    if tag:
        # construct a full new name: [stem][tag][suffix]
        # Example: "data" + "_v1" + ".csv" -> "data_v1.csv"
        base_path = base_path.with_name(f"{base_path.stem}{tag}{base_path.suffix}")

    return base_path