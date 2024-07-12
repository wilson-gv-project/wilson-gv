#!/usr/bin/env python
import os

# Directory containing SVG files
directory_path = '/home/vlew/Wilson/spectra/'
directory_pathW = 'C:/Users/vle014/OneDrive%20-%20UiT%20Office%20365/Documents/svgs/'
# Output HTML file path
output_path = '/home/vlew/Wilson/spectra/index.html'

# Retrieve list of files with their creation times
files_with_ctime = [
    (filename, os.path.getctime(os.path.join(directory_path, filename)))
    for filename in os.listdir(directory_path)
    if filename.endswith(".svg")
]
print(files_with_ctime)
# Sort files by creation time (from older to newer)
sorted_files = sorted(files_with_ctime, key=lambda x: x[1])

# Define the function to group files (example: based on filename prefix)
def group_files(files):
    groups = {}
    for filename, ctime in files:
        allnameparts = filename.split('_')
        group_name = ' '.join(allnameparts[1:5])
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append((filename, ctime))
    return groups

# Group the sorted files
grouped_files = group_files(sorted_files)
print('\ngrouped_files\n', grouped_files)

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
            grid-template-columns: repeat(auto-fill, minmax(700px, 1fr));
            gap: 15px; /* Increased gap between items */
            padding: 10px;
        }
        .grid embed  {
            width: 100%;
            height: auto;
            margin-bottom: 10px; /* Add margin bottom to each image */
        }
        .group-title {
            text-align: left;
            font-size: 1.5em;
            margin-top: 20px;
            margin-left: 1100px;
        }
    </style>
</head>
<body>
"""

# Loop through each group and generate the HTML content
for group_name, files in grouped_files.items():
    html_content += f"""
    <h2 class="group-title">{group_name}</h2>
    <div class="grid">
    """
    for filename, _ in files:
        file_path = os.path.join(directory_pathW, filename)
        print(file_path)

        file_title = 'yes'
        # <p>{file_title}</p>
        html_content += f"""
            <div>
                <embed src="{file_path}" alt="{file_title} width="70%" height="70%">
            </div>
        """
    html_content += """
    </div>
    """

# End of HTML content
html_content += """
</body>
</html>
"""

# Write the HTML content to the output file
with open(output_path, 'w') as file:
    file.write(html_content)

print('\nHTML file has been generated!')

