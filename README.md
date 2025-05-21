# OpenMCX - Minimal Image Classification Tool

OpenMCX is a lightweight, minimal desktop application for performing image classification using your own machine learning models. It provides a simple user interface to select a model, an input folder of images, and an output folder for the classified results.

## Features

* **Model Selection**: Supports common model formats (e.g., `.h5`, `.tflite`, `.pb`).
* **Label File Selection**: Requires a `.txt` file mapping class indices to human-readable names.
* **Single Input Folder**: Select one folder containing images to classify.
* **Max 50 Images**: Processes a maximum of the first 50 supported image files found in the input folder.
* **Output Folder**: Designate a folder where classified images will be (conceptually) organized into subfolders named after their predicted class.
* **Progress Display**: Shows the progress of the classification task.
* **Results List**: Displays the classification outcome for each processed image.
* **Simple UI**: A clean and straightforward interface, avoiding unnecessary complexity.
* **Path Persistence**: Remembers the last used paths for model, labels, input, and output folders for convenience.

## Requirements

* Python 3.7+
* PyQt6
* TensorFlow (or TensorFlow Lite Runtime, depending on your model)
* NumPy
* Pillow (PIL Fork - often used for image manipulation with TensorFlow/Keras)

You will also need:
* The `openmcx_main.py` script (which includes placeholders for `ClassificationThread` and file selection widgets).
* **Your own image classification model** (e.g., a `.h5` or `.tflite` file).
* **A corresponding labels file** (`.txt`), with each class name on a new line, matching the output of your model.

## Installation & Setup

1.  **Clone the repository or download the files:**
    If this were a git repository:
    ```bash
    git clone <repository_url>
    cd openmcx
    ```
    Otherwise, ensure you have `openmcx_main.py`.

2.  **Install dependencies:**
    ```bash
    pip install PyQt6 tensorflow numpy Pillow
    ```
    (If using a TFLite model exclusively, you might opt for `tflite-runtime` instead of the full TensorFlow package for a lighter setup: `pip install tflite-runtime`)

3.  **Prepare your Model and Labels:**
    * Have your trained classification model file ready.
    * Create a `labels.txt` file where each line corresponds to a class name that your model predicts. For example:
        ```
        Cat
        Dog
        Bird
        ```

## How to Run

Execute the main Python script:

```bash
python openmcx_main.py
