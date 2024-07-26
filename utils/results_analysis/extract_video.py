import imageio.v2 as imageio
import os


def download_images(name_image, chosen_run, local_path):
    """
    Downloads the images from the specified path and saves them in the local_path
    INPUT:
    - name_image: start of the name of the images to download
    - chosen_run: the run from which to download the images
    - local_path: the local path where to save the images
    """
    os.makedirs(local_path, exist_ok=True)
    images_directory_start_name = "media/images/" + name_image  # format on wandb
    count = 0
    print(f"Downloading files which start with {images_directory_start_name}...")
    for file in chosen_run.files():
        if file.name.startswith(images_directory_start_name):
            file.download(root=local_path, exist_ok=True)
            count += 1
    print(f"Total downloaded: {count} files, saved in {local_path}")


def add_files_to_images(local_path, name_image):
    """
    Add files from the local_path to a list for GIF/video creation.
    INPUT:
    - local_path: the local path where the images are saved, e.g. '../results/run'
    - name_image: start of the name of the images to download, e.g. 'class_probabilities_wrt_time_' the number of the epoch should be after the last '_'
    OUTPUT:
    - images: list of images
    - folder_path: the path to the folder where the images are saved
    """
    folder_path = os.path.join(local_path, "media", "images")  # format from wandb
    images = []
    if os.path.exists(folder_path):
        images_paths = []
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".png") and file.startswith(name_image):
                file_path = os.path.join(folder_path, file)
                images_paths.append(file_path)

        # sort the images paths by the number after "class_probabilities_wrt_time_"
        images_paths = sorted(
            images_paths, key=lambda x: int(x.split(name_image)[1].split("_")[1])
        )

        for file_path in images_paths:
            image = imageio.imread(file_path)
            if image.ndim == 3:  # Check if image is a color image (H x W x C)
                images.append(image)
            else:
                print(f"Error loading {file_path}: Image is not in expected format.")
    else:
        print(f"Folder {folder_path} does not exist. No images added.")
    return images, folder_path


def save_video(
    images_directory, images, output_filename="class_probability_wrt_time.mp4"
):
    """
    Save a list of images as a video at the specified location.
    INPUT:
    - images_directory: the directory where the images are saved
    - images: list of images
    - output_filename: the name of the video file, should end with .mp4
    OUTPUT:
    - video_path: the path to the saved video
    """
    if not output_filename.endswith(".mp4"):
        raise ValueError("output_filename should end with .mp4")
    if os.path.exists(os.path.join(images_directory)):
        writer = imageio.get_writer(
            os.path.join(images_directory, output_filename), fps=2, codec="libx264"
        )
        count = 0
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
    else:
        video_path = None
        print(f"Folder {images_directory} does not exist. No video saved.")
    return video_path
