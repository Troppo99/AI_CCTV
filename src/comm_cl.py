import re


def remove_comments_and_docstrings(source):
    """Remove comments and docstrings from a Python source file."""
    source = re.sub(re.compile(r'""".*?"""', re.DOTALL), "", source)  # Remove docstrings (""")
    source = re.sub(re.compile(r"'''.*?'''", re.DOTALL), "", source)  # Remove docstrings (''')
    source = re.sub(re.compile(r"#.*?\n"), "", source)  # Remove comments
    return source


def remove_comments(file_path):
    with open(file_path, "r") as file:
        source = file.read()

    cleaned_source = remove_comments_and_docstrings(source)

    with open(file_path, "w") as file:
        file.write(cleaned_source)


if __name__ == "__main__":
    file_path = "src/basket.py"  # Ganti dengan path file Python Anda
    remove_comments(file_path)
    print(f"Comments removed from {file_path}")
