import os

# The file we will generate for NotebookLM
OUTPUT_FILE = "codebase_for_gemini.md"

# Folders to completely ignore (we don't want to upload compiled binaries or gigabytes of data)
IGNORE_DIRS = {".git", "build", "data", "__pycache__", "_deps"}

# File extensions we actually want to share with the AI
ALLOWED_EXTENSIONS = {".cpp", ".hpp", ".h", ".py", ".txt", ".md"}

def export_codebase():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        outfile.write("# Transformer Engine C++ Codebase\n\n")
        outfile.write("This document contains the complete source code for a custom C++ Transformer Engine.\n\n")

        # Walk through the current directory
        for root, dirs, files in os.walk("."):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                ext = os.path.splitext(file)[1]
                # Special case for CMakeLists.txt which has no extension
                if ext in ALLOWED_EXTENSIONS or file == "CMakeLists.txt":
                    filepath = os.path.join(root, file)
                    
                    # Don't include the output file itself if we run it multiple times
                    if file == OUTPUT_FILE:
                        continue

                    try:
                        with open(filepath, "r", encoding="utf-8") as infile:
                            content = infile.read()

                        # Write a clear markdown header for the AI to understand the file path
                        outfile.write(f"## File: `{filepath}`\n\n")
                        
                        # Use markdown code blocks based on file type
                        lang = "cpp" if ext in {".cpp", ".hpp", ".h"} else "python" if ext == ".py" else "cmake"
                        outfile.write(f"```{lang}\n")
                        outfile.write(content)
                        outfile.write("\n```\n\n")
                        
                        print(f"Added: {filepath}")
                    except Exception as e:
                        print(f"Skipped {filepath} due to error: {e}")

    print(f"\n✅ Success! All code has been combined into: {OUTPUT_FILE}")
    print("Upload this single file to NotebookLM/Gemini!")

if __name__ == "__main__":
    export_codebase()