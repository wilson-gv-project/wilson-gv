#!/usr/bin/env python
import os

# Directory containing SVG files
# directory_path = '/home/vlew/Wilson/spectra/'
directory_path = '/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/svgs/new_specs_FOAC_anharm/'
# directory_pathW = 'C:/Users/vle014/OneDrive%20-%20UiT%20Office%20365/Documents/svgs/'
directory_pathW = 'C:/Users/vle014/OneDrive%20-%20UiT%20Office%20365/Documents/svgs/new_specs_FOAC_anharm/'
output_path = '/home/vlew/Wilson/spectra/FORM.html'

files_with_ctime = [
    (filename, os.path.getctime(os.path.join(directory_path, filename)))
    for filename in os.listdir(directory_path)
    if filename.endswith(".svg")
]
# for k in [i[0].split('/')[0].split('_el')[0] for i in files_with_ctime]:
#     print(k)

# Sort files by creation time (from older to newer)
sorted_files = sorted(files_with_ctime, key=lambda x: x[1])
# print(sorted_files)
def group_files(files):
    groups = {}
    gTitles = [i[0].split('/')[0].split('_el')[0] for i in files]
    for filename, ctime in files:
        allnameparts = filename.split('_')
        group_name = filename.split('/')[0].split('_el')[0].split('EL_')[1]
        # if 'aug' in filename:
        #     group_name = ' '.join(allnameparts[2:8])
        # else:
        #     group_name = ' '.join(allnameparts[2:7])
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append((filename, ctime))
    return groups

grouped_files = group_files(sorted_files)
print('\ngrouped_files\n', grouped_files.keys())

html_content = (f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{directory_path.split('/')[-1]}</title>"""
                +"""
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
""")

for group_name, files in grouped_files.items():
    html_content += f"""
    <h2 class="group-title">{group_name}</h2>
    <div class="grid">
    """
    for filename, _ in files:
        file_path = os.path.join(directory_pathW, filename)
        # print(file_path)

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

html_content += """
</body>
</html>
"""

with open(output_path, 'w') as file:
    file.write(html_content)

print('\nHTML file has been generated!')

