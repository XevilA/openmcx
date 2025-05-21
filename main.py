import os
import sys
import glob
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QListWidget,
    QProgressBar, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QSettings, QStandardPaths
from PyQt6.QtGui import QIcon

# For image classification (assuming these are still needed by ClassificationThread)
# import numpy as np # Uncomment if your ClassificationThread uses it
# import tensorflow as tf # Uncomment if your ClassificationThread uses it

# App constants
APP_NAME = "OpenMCX"
APP_VERSION = "1.0.1" # Incremented version
ORGANIZATION_NAME = "OpenSource"
CONFIG_PATH = os.path.join(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation),
                           ORGANIZATION_NAME, APP_NAME)

os.makedirs(CONFIG_PATH, exist_ok=True)

# --- Mock/Simplified ClassificationThread and FileWidgets ---
# If you have these files, use the actual imports.
# This is a placeholder to make the main app runnable for demonstration.

class FileSelectionWidget(QWidget):
    path_changed = pyqtSignal(str) # Signal when path changes

    def __init__(self, label_text, file_filter="", placeholder_text=""):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.label = QLabel(label_text)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(placeholder_text)
        self.path_edit.setReadOnly(True)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_file)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.path_edit)
        self.layout.addWidget(self.browse_button)
        self._file_filter = file_filter
        self._path = ""

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", self._path or QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation), self._file_filter)
        if path:
            self.set_path(path)

    def get_path(self):
        return self._path

    def set_path(self, path):
        old_path = self._path
        self._path = path
        self.path_edit.setText(path)
        if old_path != self._path:
            self.path_changed.emit(self._path)

class FolderSelectionWidget(QWidget):
    path_changed = pyqtSignal(str) # Signal when path changes

    def __init__(self, label_text, multiple=False, placeholder_text=""):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.label = QLabel(label_text)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(placeholder_text)
        self.path_edit.setReadOnly(True)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_folder)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.path_edit)
        self.layout.addWidget(self.browse_button)
        self._path = "" # For single folder selection

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder", self._path or QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation))
        if path:
            self.set_paths(path)

    def get_paths(self): # Returns a single path string
        return self._path

    def set_paths(self, path): # Expects a single path string
        old_path = self._path
        self._path = path
        self.path_edit.setText(path)
        if old_path != self._path:
            self.path_changed.emit(self._path)


class ClassificationThread(QThread):
    progress_update = pyqtSignal(int, int)
    result_update = pyqtSignal(str, str, str) # file_path, class_name, confidence
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_path, label_path, image_files, output_folder, batch_size):
        super().__init__()
        self.model_path = model_path
        self.label_path = label_path
        self.image_files = image_files
        self.output_folder = output_folder
        self.batch_size = batch_size
        self._is_running = True
        self.model = None # Placeholder for loaded model

    def run(self):
        try:
            # --- Actual ML logic should replace these checks and processing loop ---
            # Example: Check if model file exists (essential for real operation)
            if not os.path.exists(self.model_path):
                self.error.emit(f"Model file not found: {self.model_path}")
                return # Stop if critical file missing

            # Example: Check labels file
            labels = []
            if not os.path.exists(self.label_path):
                self.error.emit(f"Label file not found: {self.label_path}")
                return # Stop if critical file missing
            with open(self.label_path, 'r') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
            if not labels:
                self.error.emit("Labels file is empty or invalid.")
                return

            # --- Start processing ---
            num_files = len(self.image_files)
            for i, file_path in enumerate(self.image_files):
                if not self._is_running:
                    # No error emitted here for user cancel, main thread handles status message
                    break
                
                # Placeholder prediction logic
                # Replace with your actual model inference
                # e.g., img = tf.keras.preprocessing.image.load_img(...)
                #       predictions = self.model.predict(...)
                #       class_idx = np.argmax(predictions[0])
                #       predicted_class_name = labels[class_idx]
                #       confidence_score = float(predictions[0][class_idx])
                
                predicted_class_name = labels[i % len(labels)] # Cycle through labels for mock
                confidence_score = 0.99 - (i % 10 / 100.0) # Mock confidence

                # Simulate saving/moving file to an output subdirectory
                try:
                    output_class_folder = Path(self.output_folder) / predicted_class_name
                    output_class_folder.mkdir(parents=True, exist_ok=True)
                    # In a real app, you'd copy or move the file:
                    # import shutil
                    # shutil.copy(file_path, output_class_folder / Path(file_path).name)
                    # For placeholder, just log it conceptually
                    # print(f"Simulated: {file_path} -> {output_class_folder / Path(file_path).name}")
                except Exception as e:
                    # If a single file operation fails, you might choose to log and continue
                    # or emit an error and stop. For this example, we'll emit and stop.
                    self.error.emit(f"Error handling output for {Path(file_path).name}: {e}")
                    return # Stop processing

                self.result_update.emit(file_path, predicted_class_name, f"{confidence_score:.2f}")
                self.progress_update.emit(i + 1, num_files)
                QThread.msleep(50) # Simulate work, adjust as needed

        except Exception as e:
            # Catch-all for other unexpected errors during thread execution
            self.error.emit(f"An unexpected error occurred during classification: {str(e)}")
        finally:
            # Crucially, ensure 'finished' is always emitted so the GUI can unlock
            self.finished.emit()

    def stop(self):
        self._is_running = False

# --- End of Mock/Simplified ---

class OpenMCX(QMainWindow):
    MAX_IMAGE_FILES = 50

    def __init__(self):
        super().__init__()
        self.settings = QSettings(ORGANIZATION_NAME, APP_NAME)
        self.classification_thread = None
        self.batch_size = 16
        self.was_cancelled_or_error = False # Flag for thread completion state

        self.init_ui()
        self.load_settings()

    def init_ui(self):
        self.setWindowTitle(f"{APP_NAME} - v{APP_VERSION}")
        self.setMinimumSize(600, 550) # Increased height slightly for better layout

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # --- Input Selection ---
        input_group = QFrame()
        input_group.setFrameShape(QFrame.Shape.StyledPanel)
        input_layout = QVBoxLayout(input_group)
        input_layout.addWidget(QLabel("<b>1. Select Inputs</b>"))
        self.model_file_widget = FileSelectionWidget("Model File:", "Model Files (*.h5 *.tflite *.pb);;All Files (*)", "Select your ML model file...")
        self.label_file_widget = FileSelectionWidget("Labels File:", "Text Files (*.txt);;All Files (*)", "Select your labels file...")
        self.input_folder_widget = FolderSelectionWidget("Input Folder:", placeholder_text=f"Select folder with images (max {self.MAX_IMAGE_FILES})...")
        input_layout.addWidget(self.model_file_widget)
        input_layout.addWidget(self.label_file_widget)
        input_layout.addWidget(self.input_folder_widget)
        main_layout.addWidget(input_group)

        # --- Output Selection ---
        output_group = QFrame()
        output_group.setFrameShape(QFrame.Shape.StyledPanel)
        output_layout = QVBoxLayout(output_group)
        output_layout.addWidget(QLabel("<b>2. Select Output</b>"))
        self.output_folder_widget = FolderSelectionWidget("Output Folder:", placeholder_text="Select folder for classified images...")
        output_layout.addWidget(self.output_folder_widget)
        main_layout.addWidget(output_group)

        # --- Actions ---
        action_group = QFrame()
        action_group.setFrameShape(QFrame.Shape.StyledPanel)
        action_layout = QVBoxLayout(action_group)
        action_layout.addWidget(QLabel("<b>3. Process</b>"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v/%m (%p%)")
        action_layout.addWidget(self.progress_bar)
        button_layout = QHBoxLayout()
        self.classify_button = QPushButton("Start Classification")
        self.classify_button.setMinimumHeight(35)
        self.classify_button.clicked.connect(self.start_classification)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(35)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_classification)
        button_layout.addWidget(self.classify_button)
        button_layout.addWidget(self.cancel_button)
        action_layout.addLayout(button_layout)
        main_layout.addWidget(action_group)

        # --- Results ---
        results_group = QFrame()
        results_group.setFrameShape(QFrame.Shape.StyledPanel)
        results_layout = QVBoxLayout(results_group)
        results_layout.addWidget(QLabel("<b>4. Results</b>"))
        self.results_list = QListWidget()
        self.results_list.setMinimumHeight(100)
        self.clear_results_button = QPushButton("Clear Results")
        self.clear_results_button.clicked.connect(self.clear_results)
        results_layout.addWidget(self.results_list)
        results_layout.addWidget(self.clear_results_button, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(results_group)
        
        main_layout.addStretch()
        self.statusBar().showMessage("Ready")
        self.center_on_screen()

    def center_on_screen(self):
        try:
            screen_geometry = self.screen().availableGeometry()
            x = (screen_geometry.width() - self.width()) // 2
            y = (screen_geometry.height() - self.height()) // 2
            self.move(x, y)
        except Exception as e:
            print(f"Could not center window: {e}")


    def load_settings(self):
        self.model_file_widget.set_path(self.settings.value("paths/model", ""))
        self.label_file_widget.set_path(self.settings.value("paths/labels", ""))
        self.input_folder_widget.set_paths(self.settings.value("paths/input_folder", ""))
        self.output_folder_widget.set_paths(self.settings.value("paths/output_folder", ""))

    def save_settings(self):
        if self.model_file_widget.get_path(): self.settings.setValue("paths/model", self.model_file_widget.get_path())
        if self.label_file_widget.get_path(): self.settings.setValue("paths/labels", self.label_file_widget.get_path())
        if self.input_folder_widget.get_paths(): self.settings.setValue("paths/input_folder", self.input_folder_widget.get_paths())
        if self.output_folder_widget.get_paths(): self.settings.setValue("paths/output_folder", self.output_folder_widget.get_paths())

    def start_classification(self):
        model_path = self.model_file_widget.get_path()
        label_path = self.label_file_widget.get_path()
        input_folder = self.input_folder_widget.get_paths()
        output_folder = self.output_folder_widget.get_paths()

        if not all([model_path, label_path, input_folder, output_folder]):
            QMessageBox.warning(self, "Input Error", "All fields (Model, Labels, Input, Output) are required.")
            return

        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp"] # Added webp
        all_image_files = []
        for ext in image_extensions:
            all_image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        
        if not all_image_files:
            QMessageBox.information(self, "No Images", f"No supported image files found in {input_folder}.")
            return

        image_files_to_process = sorted(all_image_files)[:self.MAX_IMAGE_FILES] # Sort for consistent order
        if len(all_image_files) > self.MAX_IMAGE_FILES:
            QMessageBox.information(self, "Image Limit", f"Found {len(all_image_files)} images. Processing the first {self.MAX_IMAGE_FILES} (alphabetically).")

        self.save_settings()
        self.clear_results()
        self.statusBar().showMessage("Starting classification...")
        self.was_cancelled_or_error = False # Reset flag

        self.classification_thread = ClassificationThread(model_path, label_path, image_files_to_process, output_folder, self.batch_size)
        self.classification_thread.progress_update.connect(self.update_progress)
        self.classification_thread.result_update.connect(self.add_result)
        self.classification_thread.finished.connect(self.classification_finished)
        self.classification_thread.error.connect(self.show_error)

        self.classify_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(image_files_to_process))
        self.classification_thread.start()

    def cancel_classification(self):
        if self.classification_thread and self.classification_thread.isRunning():
            self.statusBar().showMessage("Canceling classification...")
            self.was_cancelled_or_error = True
            self.classification_thread.stop()
            # classification_finished will handle other UI updates

    def update_progress(self, value, maximum):
        if maximum > 0 : # Ensure maximum is positive to prevent division by zero for format string
             self.progress_bar.setMaximum(maximum)
             self.progress_bar.setValue(value)
        else: # handles case of 0 images processed initially or empty list
             self.progress_bar.setMaximum(1) # Avoid division by zero if max is 0
             self.progress_bar.setValue(0)
        self.statusBar().showMessage(f"Processing: {value}/{maximum}")


    def add_result(self, file_path, class_name, confidence):
        item_text = f"{os.path.basename(file_path)} → {class_name} (Conf: {confidence})"
        self.results_list.addItem(item_text)
        self.results_list.scrollToBottom()

    def classification_finished(self):
        current_status = self.statusBar().currentMessage()

        if self.was_cancelled_or_error:
            if current_status == "Canceling classification...": # Set final cancel message
                self.statusBar().showMessage("Classification canceled by user.")
            # If it was an error, show_error already set the message.
        else:
            self.statusBar().showMessage("Classification complete.")
            if self.progress_bar.value() == 0 and self.progress_bar.maximum() > 0: # No items processed but files were found
                 self.statusBar().showMessage("Classification complete. No items were processed (check model/labels).")
            elif self.progress_bar.maximum() == 0 : # No image files were passed to thread initially
                 self.statusBar().showMessage("No images were found to process.")


        self.classify_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        # No specific operations on self.classification_thread needed before setting to None
        # as its signals have been processed.
        self.classification_thread = None
        # self.was_cancelled_or_error is reset in start_classification

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.statusBar().showMessage(f"Error: {error_message}")
        self.was_cancelled_or_error = True
        
        # Ensure UI is reset if an error occurs, even if finished signal also fires
        self.classify_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        # Do not set self.classification_thread to None here;
        # let classification_finished handle it as the 'finished' signal will still be emitted.

    def clear_results(self):
        self.results_list.clear()
        self.progress_bar.setValue(0)
        if not (self.classification_thread and self.classification_thread.isRunning()):
            self.statusBar().showMessage("Results cleared. Ready.")

    def closeEvent(self, event):
        self.save_settings()
        thread_to_wait_for = self.classification_thread # Keep a local reference
        if thread_to_wait_for and thread_to_wait_for.isRunning():
            self.statusBar().showMessage("Closing: Stopping classification...")
            self.was_cancelled_or_error = True # Ensure status isn't overwritten
            thread_to_wait_for.stop()
            if not thread_to_wait_for.wait(2500): # wait returns true if thread finished; increased timeout
                print("Warning: Classification thread did not stop cleanly on close.")
            # else:
            # print("Classification thread stopped on close.")
        event.accept()

def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    main_window = OpenMCX()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
