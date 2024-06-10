import imageio.v2 as imageio
import os 

def download_images(path, name_image, chosen_run, local_path):
    """
    download_images: Downloads the images from the specified path and saves them in the local_path
    INPUT: 
    - path: path to the images in the wandb storage
    - name_image: start of the name of the images to download
    - chosen_run: the run from which to download the images
    - local_path: the local path where to save the images
    """
    # create the directory
    os.makedirs(local_path, exist_ok=True)
    # List and download the files from the specified path
    images_directory_start_name = path + name_image
    count = 0
    print(f"Downloading files which start with {images_directory_start_name}...")
    for file in chosen_run.files():
        if file.name.startswith(images_directory_start_name):
            file.download(root=local_path, exist_ok=True)
            count+=1
            print(f"Downloaded {file.name}")
    print(f"Total downloaded: {count} files.")
    
    
def add_files_to_images(folder_path, name_image):
    """Add files from a folder to a list for GIF creation."""
    images = []
    images_paths = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.png') and file.startswith(name_image):
            file_path = os.path.join(folder_path, file)
            images_paths.append(file_path)
            
    # sort the images paths by then number after "class_probabilities_wrt_time_"
    images_paths = sorted(images_paths, key=lambda x: int(x.split(name_image)[1].split("_")[1]))
    print("imags_paths: ", images_paths)
    
    for file_path in images_paths:
        image = imageio.imread(file_path)
        if image.ndim == 3:  # Check if image is a color image (H x W x C)
            images.append(image)
        else:
            print(f"Error loading {file_path}: Image is not in expected format.")
            
    return images


def save_gif(images_directory, images, filename='class_probability_wrt_time.gif', duration=1.):
    """Save a list of images as a GIF at the specified location."""
    gif_path = os.path.join(images_directory, filename)
    imageio.mimsave(gif_path, images, duration=duration)  # duration controls the timing between frames in seconds
    print(f"GIF saved at {gif_path}")
    return gif_path


def save_video(images_directory, images, output_filename='class_probability_wrt_time.mp4'):
    writer = imageio.get_writer(os.path.join(images_directory, output_filename), fps=2, codec='libx264') 
    count = 0  # To count images processed
    for image in images:
        if image is not None:
            writer.append_data(image)
            count += 1
        else:
            print(f"Failed to read image.")
    writer.close()
    print(f"{count} images added to the video.")
    video_path = os.path.join(images_directory, output_filename)
    print(f"Video saved at {video_path}")
    return video_path