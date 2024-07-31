from pathlib import Path

def root_project(cd_to: str = None) -> Path:
    cwd = Path.cwd()
    while cwd.name != 'florentino':
        cwd = cwd.parent
    if cd_to:
        cwd /= cd_to
    return cwd

if __name__ == '__main__':
    print(root_project())
    print(root_project('datasets'))
