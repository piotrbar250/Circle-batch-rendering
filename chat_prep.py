# Import required library
import os

# Function to read filenames and comments, then write to a single file
def create_combined_file_with_comments(input_filename, output_filename):
    try:
        # Open the file with list of filenames and comments
        with open(input_filename, 'r') as file_list:
            # Read all lines
            lines = file_list.readlines()

        # Split the lines into filenames and comments based on the '---' delimiter
        delimiter_index = lines.index('---\n')
        filenames = lines[:delimiter_index]
        comments = lines[delimiter_index + 1:]

        # Open the output file for writing
        with open(output_filename, 'w') as output_file:
            # Write comments first
            for comment in comments:
                output_file.write(comment)

            # Add an extra line after comments
            output_file.write('\n')

            # Iterate over each filename
            for filename in filenames:
                filename = filename.strip()  # Remove any leading/trailing whitespaces
                if os.path.exists(filename):
                    # Write the filename and its content
                    output_file.write(f'{filename}\n\n')
                    with open(filename, 'r') as file:
                        output_file.write(file.read() + '\n\n')
                else:
                    print(f"File not found: {filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
create_combined_file_with_comments('chat_prep.txt', 'combined_output.txt')
