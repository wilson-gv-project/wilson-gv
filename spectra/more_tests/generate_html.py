#!/usr/bin/env python
import os

# Directory containing SVG files
directory_path = './svgs'
# Output HTML file path
output_path = 'index.html'

# Retrieve list of files with their creation times
files_with_ctime = [(filename, os.path.getctime(os.path.join(directory_path, filename)))
                    for filename in os.listdir(directory_path)
                    if filename.endswith(".svg")]
# print(files_with_ctime)
# Sort files by creation time (from older to newer)
sorted_files = sorted(files_with_ctime, key=lambda x: x[1])
# print(sorted_files)

# Start of HTML content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVG Gallery</title>
    <style>
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            padding: 10px;
        }
        .grid img {
            width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <div class="grid">
"""

# Loop through each sorted file
for filename, _ in sorted_files:
    file_path = os.path.join(directory_path, filename)
    file_title = os.path.splitext(filename)[0]
    html_content += f"""
        <div>
            <img src="{file_path}" alt="{file_title}">
            <p>{file_title}</p>
        </div>
"""

# End of HTML content
html_content += """
    </div>
</body>
</html>
"""

# Write the HTML content to the output file
with open(output_path, 'w') as file:
    file.write(html_content)

print('HTML file has been generated!')