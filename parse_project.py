import os

def parse_project(project_path, output_file="project_parsed.txt", ignored_files=None, ignored_extensions=None, ignored_dirs=None):
    """
    Parses a project directory, concatenating the contents of specified files into a single output file.

    Args:
        project_path (str): The path to the project directory.
        output_file (str, optional): The name of the file to save the parsed project to. Defaults to "project_parsed.txt".
        ignored_files (list, optional): A list of specific file names to ignore. Defaults to None.
        ignored_extensions (list, optional): A list of file extensions to ignore. Defaults to None.
        ignored_dirs (list, optional): A list of directory names to ignore. Defaults to None.
    """
    # --- Updated Ignore Lists Based on Your Project Structure ---
    if ignored_files is None:
        ignored_files = [
            '__init__.py',
            '.gitignore',
            'client_secret_396977883868-bi2dp352pvs09ms6f88n1vbp1gplh925.apps.googleusercontent.com.json',
            'new-service-account-key.json',
            'lifecycle.json',
            'requirements.txt',
            'Dockerfile',
            'parse_project.py', # To avoid the script parsing itself
            '.DS_Store' # Ignoring macOS system files
        ]
    if ignored_extensions is None:
        # Ignoring .json files as they appear to be keys/config
        ignored_extensions = ['.pyc', '.git', '.idea', '.vscode', '.json']
    if ignored_dirs is None:
        ignored_dirs = [
            '__pycache__',
            '.git',
            '.idea',
            '.vscode',
            'venv', # Ignoring the virtual environment
            'static' # Ignoring the entire static folder
        ]
    # --- End of Updates ---

    with open(output_file, "w", encoding="utf-8") as outfile:
        for dirpath, dirnames, filenames in os.walk(project_path):
            # Remove ignored directories from traversal
            dirnames[:] = [d for d in dirnames if d not in ignored_dirs]

            for filename in filenames:
                # Check if the file should be ignored
                if filename not in ignored_files and not any(filename.endswith(ext) for ext in ignored_extensions):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                            # Adding a clear header for each file
                            outfile.write("-" * 50 + "\n")
                            outfile.write(f"# File: {os.path.relpath(file_path, project_path)}\n")
                            outfile.write("-" * 50 + "\n\n")
                            outfile.write(infile.read())
                            outfile.write("\n\n")
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

    print(f"Project parsed successfully. Output saved to {output_file}")

if __name__ == "__main__":
    # Get the project path from the user. Defaults to the current directory.
    project_path = input("Enter the path to your project directory (or press Enter for current directory): ")
    if not project_path:
        project_path = "." # Use current directory if input is empty
    
    # Check if the path exists
    if os.path.isdir(project_path):
        parse_project(project_path)
    else:
        print("The specified path does not exist or is not a directory.")
