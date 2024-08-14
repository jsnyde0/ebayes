#!/bin/bash

# Example usage: ./copy_codebase.sh ./a_core/ ./b_mmm/ ./templates/

# Check if at least one path argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide one or more relative paths as arguments."
    exit 1
fi

# Function to process a single directory
process_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "The specified directory does not exist: $dir"
        return
    fi

    find "$dir" -type f \( -name "*.html" -o -name "*.py" \) \
        -not -path "*/venv/*" \
        -not -path "*/__pycache__/*" \
        -not -path "*/migrations/*" \
        -print0 | while IFS= read -r -d '' file; do
        echo "---" >> "$temp_file"
        echo "$file" >> "$temp_file"
        echo "---" >> "$temp_file"
        cat "$file" >> "$temp_file"
        echo "" >> "$temp_file"
    done
}

# Create a temporary file
temp_file=$(mktemp)

try_copy_to_clipboard() {
    if command -v xclip &> /dev/null; then
        xclip -selection clipboard < "$temp_file"
        echo "Content copied to clipboard using xclip."
    elif command -v clip.exe &> /dev/null; then
        cat "$temp_file" | clip.exe
        echo "Content copied to clipboard using clip.exe (WSL)."
    else
        echo "Unable to copy to clipboard. Please install xclip or use WSL with clip.exe available."
        return 1
    fi
}

# Process each directory provided as an argument
for dir in "$@"; do
    process_directory "$dir"
done

# Count tokens
word_count=$(wc -w < "$temp_file")
token_count=$((word_count * 3 / 4))

# Try to copy to clipboard
if try_copy_to_clipboard; then
    echo "Files from all specified directories containing ~$token_count tokens have been copied to clipboard."
else
    echo "Content was not copied to clipboard. You can find it in $temp_file"
fi

# Clean up temporary file
rm -f "$temp_file"