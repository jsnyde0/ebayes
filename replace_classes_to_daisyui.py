import re

# Define the replacements
replacements = {
    r'\bbg-white\b': 'bg-base-100',
    r'\bbg-gray-50\b': 'bg-base-200',
    r'\bbg-gray-100\b': 'bg-base-200',
    r'\bbg-gray-200\b': 'bg-base-300',
    r'\bbg-gray-300\b': 'bg-neutral',
    r'\bbg-gray-400\b': 'bg-neutral-focus',
    r'\bbg-gray-500\b': 'bg-neutral-content',
    r'\bbg-gray-600\b': 'bg-neutral-content',
    r'\bbg-gray-700\b': 'bg-base-300',
    r'\bbg-gray-800\b': 'bg-base-200',
    r'\bbg-gray-900\b': 'bg-base-100',
    r'\bbg-blue-600\b': 'bg-primary',
    r'\bbg-blue-700\b': 'bg-primary-focus',
    r'\bbg-red-600\b': 'bg-error',
    r'\bbg-green-600\b': 'bg-success',
    r'\bbg-yellow-600\b': 'bg-warning',
    r'\btext-gray-500\b': 'text-base-content',
    r'\btext-gray-600\b': 'text-base-content',
    r'\btext-gray-700\b': 'text-base-content',
    r'\btext-gray-800\b': 'text-base-content',
    r'\btext-gray-900\b': 'text-base-content',
    r'\btext-white\b': 'text-base-100',
    r'\btext-blue-600\b': 'text-primary',
    r'\btext-red-600\b': 'text-error',
    r'\btext-green-600\b': 'text-success',
    r'\btext-yellow-600\b': 'text-warning',
    r'\bborder-gray-200\b': 'border-base-300',
    r'\bborder-gray-300\b': 'border-base-content',
    r'\bborder-blue-600\b': 'border-primary',
    r'\bborder-red-600\b': 'border-error',
    r'\bborder-green-600\b': 'border-success',
    r'\bborder-yellow-600\b': 'border-warning',
    r'\bhover:bg-gray-100\b': 'hover:bg-base-200',
    r'\bhover:bg-gray-600\b': 'hover:bg-neutral',
    r'\bhover:bg-blue-700\b': 'hover:bg-primary-focus',
    r'\bhover:text-gray-900\b': 'hover:text-base-content',
    r'\bfocus:ring-gray-200\b': 'focus:ring-base-300',
    r'\bfocus:ring-blue-300\b': 'focus:ring-primary-focus',
    r'\bplaceholder-gray-400\b': 'placeholder-base-content',
    r'\bdark:bg-gray-800\b': 'dark:bg-base-300',
    r'\bdark:bg-gray-700\b': 'dark:bg-base-200',
    r'\bdark:text-white\b': 'dark:text-base-content',
    r'\bdark:border-gray-600\b': 'dark:border-base-content',
    r'\bdark:placeholder-gray-400\b': 'dark:placeholder-base-content',
    r'\bdark:hover:bg-gray-700\b': 'dark:hover:bg-base-200',
}

def replace_classes(content):
    for old, new in replacements.items():
        content = re.sub(old, new, content)
    return content

# Read the input file
input_file = 'templates/mmm/home.html'
try:
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
except UnicodeDecodeError:
    # If UTF-8 fails, try with ISO-8859-1 encoding
    with open(input_file, 'r', encoding='iso-8859-1') as file:
        content = file.read()

# Perform replacements
modified_content = replace_classes(content)

# Write the modified content to a new file
output_file = 'templates/mmm/home_daisyui.html'
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(modified_content)

print(f"Replacements complete. Modified content saved to {output_file}")