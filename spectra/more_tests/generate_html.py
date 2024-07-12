#!/usr/bin/env python
# import os
#
# # Directory containing SVG files
# directory_path = './svgs'
# # Output HTML file path
# output_path = 'index.html'
#
# # Retrieve list of files with their creation times
# files_with_ctime = [(filename, os.path.getctime(os.path.join(directory_path, filename)))
#                     for filename in os.listdir(directory_path)
#                     if filename.endswith(".svg") and 'w1mw2' in filename]
# files_with_ctime = sorted([i for i in files_with_ctime if 'elT_mechF' in i[0]])
# print(files_with_ctime)
#
# # Sort files by creation time (from older to newer)
# sorted_files = sorted(files_with_ctime, key=lambda x: x[1])
# # print(sorted_files)
#
# # Start of HTML content
# html_content = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>SVG Gallery</title>
#     <style>
#         .grid {
#             display: grid;
#             grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
#             gap: 10px;
#             padding: 10px;
#         }
#         .grid img {
#             width: 100%;
#             height: auto;
#         }
#     </style>
# </head>
# <body>
#     <div class="grid">
# """
#
# # Loop through each sorted file
# for filename, _ in sorted_files:
#     file_path = os.path.join(directory_path, filename)
#     file_title = os.path.splitext(filename)[0]
#     html_content += f"""
#         <div>
#             <img src="{file_path}" alt="{file_title}">
#             <p>{file_title}</p>
#         </div>
# """
#
# # End of HTML content
# html_content += """
#     </div>
# </body>
# </html>
# """
#
# # Write the HTML content to the output file
# with open(output_path, 'w') as file:
#     file.write(html_content)
#
# print('\nHTML file has been generated!')

#!/usr/bin/env python
import os

# Directory containing SVG files
directory_path = '/home/vlew/Wilson/spectra'
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
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 35px; /* Increased gap between items */
            padding: 10px;
        }
        .grid img {
            width: 100%;
            height: auto;
            margin-bottom: 10px; /* Add margin bottom to each image */
        }
        .group-title {
            text-align: center;
            font-size: 1.5em;
            margin-top: 20px;
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
        file_path = os.path.join(directory_path, filename)
        # file_title = os.path.splitext(filename)[0]
        if 'gamma' in file_path:
            file_title = '_'.join(os.path.splitext(filename)[0].split('_')[:4])
            # files_with_ctime = [(, i[1]) for i in files_with_ctime if 'gamma' in i[0]]
        else:
            file_title = os.path.splitext(filename)[0]
        html_content += f"""
            <div>
                <img src="{file_path}" alt="{file_title}">
                <p>{file_title}</p>
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

