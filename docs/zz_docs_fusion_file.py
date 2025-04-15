##############################################
# IMPORT LIBRARIES ###########################
##############################################

"""
Libraries and packages used in this script.
"""
import os  # Version: N/A (built-in)

###############################################
# CONSTANTS & PARAMETERS ######################
###############################################

"""
Constants and parameters used in this script.
"""
OUTPUT_FILE = "zz_fused.md"

##############################################
# FUNCTION DEFINITIONS #######################
##############################################

"""
Functions must adhere to the following structure:

1) Use docstrings to describe the function's purpose, parameters, and return values.
2) Be named according to their functionality.
"""

def fuse_markdown_files(root_dir, output_file):
    """
    Fuses all .md files in the given directory and its subdirectories into a single .md file.

    Args:
        root_dir (str): The root directory to start searching for .md files.
        output_file (str): The name of the output file.

    Returns:
        tuple: A tuple containing the number of files fused and the total word count.
    """
    all_md_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                all_md_files.append(os.path.join(root, file))

    num_files = len(all_md_files)
    total_words = 0
    with open(output_file, "w") as outfile:
        for md_file in all_md_files:
            try:
                with open(md_file, "r") as infile:
                    content = infile.read()
                    words = content.split()
                    total_words += len(words)
                    outfile.write(f"[{md_file}]:\n\n")
                    outfile.write(content)
                    outfile.write("\n\n")
            except Exception as e:
                print(f"Error processing file {md_file}: {e}")
                num_files -=1 # Discount the file if it fails to read
                continue

    return num_files, total_words

##################################################################################
# MAIN PROGRAM ###################################################################
##################################################################################

"""
Main program to fuse markdown files and output statistics.
"""

if __name__ == "__main__":
    #########################################
    # FUSE MARKDOWN FILES ###################
    #########################################

    num_files, total_words = fuse_markdown_files(".", OUTPUT_FILE)

    #########################################
    # OUTPUT RESULTS ########################
    #########################################

    print(f"Number of files fused: {num_files}")
    print(f"Total words: {total_words}")
    print(f"Fused file name: {OUTPUT_FILE}")