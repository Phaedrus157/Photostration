import os
import glob

def get_render_image_path():
    """
    Automatically finds and returns the path to the first valid image file in the PhotRend folder.
    Supports TIFF, PNG, and JPEG formats, prioritizing TIFF for quality analysis.
    
    Returns:
        str: Path to the image file, or None if no valid image is found
        
    Raises:
        FileNotFoundError: If PhotRend folder doesn't exist
        ValueError: If no valid image files are found in PhotRend folder
    """
    render_folder = "PhotRend"
    
    # Check if PhotRend folder exists
    if not os.path.exists(render_folder):
        raise FileNotFoundError(f"PhotRend folder not found. Please create the '{render_folder}' directory.")
    
    # Define supported image extensions in priority order (TIFF preferred for quality)
    extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
    
    # Search for image files in order of preference
    for ext in extensions:
        pattern = os.path.join(render_folder, ext)
        matches = glob.glob(pattern)
        if matches:
            # Return the first match for this extension
            image_path = matches[0]
            print(f"Found rendering image: {image_path}")
            return image_path
    
    # If no images found, provide helpful error message
    available_files = os.listdir(render_folder)
    if available_files:
        print(f"Files in {render_folder}: {available_files}")
        raise ValueError(f"No valid image files found in '{render_folder}' folder. Supported formats: TIFF, PNG, JPEG")
    else:
        raise ValueError(f"'{render_folder}' folder is empty. Please add an image file for analysis.")

def get_render_image_info():
    """
    Gets the rendering image path and basic information.
    
    Returns:
        tuple: (image_path, filename, extension)
    """
    try:
        image_path = get_render_image_path()
        filename = os.path.basename(image_path)
        extension = os.path.splitext(filename)[1].lower()
        return image_path, filename, extension
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return None, None, None

if __name__ == "__main__":
    # Test the function
    try:
        path, name, ext = get_render_image_info()
        if path:
            print(f"Rendering Image Path: {path}")
            print(f"Filename: {name}")  
            print(f"Format: {ext}")
    except Exception as e:
        print(f"Error: {e}")