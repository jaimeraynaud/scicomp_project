use std::fs;
use std::io;

// Function to delete all files in a nested folder
pub fn delete_files_in_folder(folder_path: &str) -> io::Result<()> {
    // Iterate over the entries in the folder
    for entry in fs::read_dir(folder_path)? {
        let entry = entry?;
        let path = entry.path();

        // Check if the entry is a file
        if path.is_file() {
            // Delete the file
            fs::remove_file(path)?;
        } else if path.is_dir() {
            // If the entry is a directory, recursively delete files inside it
            delete_files_in_folder(path.to_str().unwrap())?;
        }
    }

    Ok(())
}