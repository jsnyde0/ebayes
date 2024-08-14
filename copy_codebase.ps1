# Example usage: .\copy_codebase.ps1 .\a_tasks\ .\a_core\ .\templates\
# Check if at least one path argument is provided
if ($args.Count -eq 0) {
    Write-Host "Please provide one or more relative paths as arguments."
    exit 1
}

# Function to process a single directory
function Process-Directory($dir) {
    if (-not (Test-Path $dir -PathType Container)) {
        Write-Host "The specified directory does not exist: $dir"
        return
    }

    Get-ChildItem -Path $dir -Recurse -File | 
        Where-Object { 
            $_.Extension -in ".html", ".py" -and 
            $_.FullName -notmatch "\\venv\\" -and 
            $_.FullName -notmatch "\\__pycache__\\" -and 
            $_.FullName -notmatch "\\migrations\\"
        } | 
        ForEach-Object {
            "---" | Add-Content $tempFile
            $_.FullName | Add-Content $tempFile
            "---" | Add-Content $tempFile
            Get-Content $_.FullName | Add-Content $tempFile
            "" | Add-Content $tempFile
        }
}

# Create a temporary file
$tempFile = [System.IO.Path]::GetTempFileName()

try {
    # Process each directory provided as an argument
    foreach ($dir in $args) {
        Process-Directory $dir
    }

    # Count tokens
    $content = Get-Content $tempFile -Raw
    $wordCount = ($content | Measure-Object -Word).Words
    $tokenCount = [int]($wordCount * 0.75)

    # Copy to clipboard using Windows Forms
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.Clipboard]::SetText($content)

    # Print token count message
    Write-Host "Files from all specified directories containing ~$tokenCount tokens have been copied to clipboard."

    # Verify clipboard content
    $clipboardContent = [System.Windows.Forms.Clipboard]::GetText()
    if ($clipboardContent -eq $content) {
        Write-Host "Content successfully copied to clipboard."
    } else {
        Write-Host "Warning: Clipboard content may not have been updated correctly."
    }
}
catch {
    Write-Host "An error occurred: $_"
}
finally {
    # Clean up temporary file
    if (Test-Path $tempFile) {
        Remove-Item $tempFile
    }
}