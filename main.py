import os
import sys
import glob
from pathlib import Path
import time # For debugging, can be removed
import shutil # Needed for file copying

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QListWidget,
    QProgressBar, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QSettings, QStandardPaths
from PyQt6.QtGui import QIcon

# App constants
APP_NAME = "OpenMCX"
APP_VERSION = "1.0.3" # Incremented version for this fix
ORGANIZATION_NAME = "OpenSource"
CONFIG_PATH = os.path.join(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation),
                           ORGANIZATION_NAME, APP_NAME)

os.makedirs(CONFIG_PATH, exist_ok=True)

# --- File Selection Widgets (no changes from previous stable version) ---
class FileSelectionWidget(QWidget):
    path_changed = pyqtSignal(str)

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
        start_dir = self._path or QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation)
        path, _ = QFileDialog.getOpenFileName(self, "Select File", start_dir, self._file_filter)
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
    path_changed = pyqtSignal(str)

    def __init__(self, label_text, multiple=False, placeholder_text=""): # multiple not used for minimal
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
        self._path = ""

    def _browse_folder(self):
        start_dir = self._path or QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation)
        path = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if path:
            self.set_paths(path)

    def get_paths(self):
        return self._path

    def set_paths(self, path):
        old_path = self._path
        self._path = path
        self.path_edit.setText(path)
        if old_path != self._path:
            self.path_changed.emit(self._path)


# --- Classification Thread (Updated with file copy) ---
class ClassificationThread(QThread):
    progress_update = pyqtSignal(int, int)
    result_update = pyqtSignal(str, str, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_path, label_path, image_files, output_folder, batch_size):
        super().__init__()
        self.model_path = model_path
        self.label_path = label_path
        self.image_files = image_files
        self.output_folder = output_folder
        # self.batch_size = batch_size # Not used in placeholder
        self._is_running = True
        # self.model = None # Placeholder for loaded model

    def run(self):
        try:
            if not self._is_running: return

            if not os.path.exists(self.model_path):
                self.error.emit(f"Model file not found: {self.model_path}")
                return
            # print(f"Thread: Attempting to load model: {self.model_path}")
            # self.model = tf.keras.models.load_model(self.model_path) # Example load

            labels = []
            if not os.path.exists(self.label_path):
                self.error.emit(f"Label file not found: {self.label_path}")
                return
            with open(self.label_path, 'r', encoding='utf-8') as f: # Added encoding for labels file
                labels = [line.strip() for line in f.readlines() if line.strip()]
            if not labels:
                self.error.emit("Labels file is empty or could not be read.")
                return
            # print(f"Thread: Labels loaded: {labels}")

            num_files = len(self.image_files)
            if num_files == 0:
                self.progress_update.emit(0,0)
                return

            for i, file_path in enumerate(self.image_files):
                if not self._is_running:
                    # print("Thread: Processing loop interrupted by stop signal.")
                    break

                predicted_class_name = labels[i % len(labels)]
                confidence_score = 0.99 - (i % 10 / 100.0)

                try:
                    output_class_folder = Path(self.output_folder) / predicted_class_name
                    output_class_folder.mkdir(parents=True, exist_ok=True)

                    # --- ACTUAL FILE COPY OPERATION ---
                    destination_file_path = output_class_folder / Path(file_path).name
                    shutil.copy2(file_path, destination_file_path)
                    # print(f"Thread: Copied {file_path} to {destination_file_path}")
                    # --- END FILE COPY ---

                except Exception as e_io:
                    self.error.emit(f"Error copying file '{Path(file_path).name}' to output: {e_io}")
                    return

                self.result_update.emit(file_path, predicted_class_name, f"{confidence_score:.2f}")
                self.progress_update.emit(i + 1, num_files)
                QThread.msleep(75)

            # print("Thread: Processing loop completed or broken.")

        except Exception as e_thread_run:
            # print(f"Thread: Unexpected error in run(): {e_thread_run}")
            self.error.emit(f"Unexpected error in classification thread: {str(e_thread_run)}")
        finally:
            # print("Thread: run() method finished or errored. Emitting finished signal.")
            self.finished.emit()

    def stop(self):
        # print("Thread: stop() called.")
        self._is_running = False

# --- Main Application Window (no changes from previous stable version) ---
class OpenMCX(QMainWindow):
    MAX_IMAGE_FILES = 50

    def __init__(self):
        super().__init__()
        self.settings = QSettings(ORGANIZATION_NAME, APP_NAME)
        self.classification_thread = None
        self.was_cancelled_or_error_during_processing = False

        self.init_ui()
        self.load_settings()

    def init_ui(self):
        self.setWindowTitle(f"{APP_NAME} - v{APP_VERSION}")
        self.setMinimumSize(600, 550)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15); main_layout.setContentsMargins(15,15,15,15)

        input_group = QFrame(); input_group.setFrameShape(QFrame.Shape.StyledPanel)
        input_layout = QVBoxLayout(input_group); input_layout.addWidget(QLabel("<b>1. Select Inputs</b>"))
        self.model_file_widget = FileSelectionWidget("Model File:", "Model Files (*.h5 *.tflite *.pb);;All Files (*)", "Select your ML model file...")
        self.label_file_widget = FileSelectionWidget("Labels File:", "Text Files (*.txt);;All Files (*)", "Select your labels file...")
        self.input_folder_widget = FolderSelectionWidget("Input Folder:", placeholder_text=f"Select folder (max {self.MAX_IMAGE_FILES} images)...")
        input_layout.addWidget(self.model_file_widget); input_layout.addWidget(self.label_file_widget); input_layout.addWidget(self.input_folder_widget)
        main_layout.addWidget(input_group)

        output_group = QFrame(); output_group.setFrameShape(QFrame.Shape.StyledPanel)
        output_layout = QVBoxLayout(output_group); output_layout.addWidget(QLabel("<b>2. Select Output</b>"))
        self.output_folder_widget = FolderSelectionWidget("Output Folder:", placeholder_text="Select folder for classified images...")
        output_layout.addWidget(self.output_folder_widget)
        main_layout.addWidget(output_group)

        action_group = QFrame(); action_group.setFrameShape(QFrame.Shape.StyledPanel)
        action_layout = QVBoxLayout(action_group); action_layout.addWidget(QLabel("<b>3. Process</b>"))
        self.progress_bar = QProgressBar(); self.progress_bar.setTextVisible(True); self.progress_bar.setFormat("%v/%m (%p%)")
        action_layout.addWidget(self.progress_bar)
        button_layout = QHBoxLayout()
        self.classify_button = QPushButton("Start Classification"); self.classify_button.setMinimumHeight(35)
        self.classify_button.clicked.connect(self.start_classification)
        self.cancel_button = QPushButton("Cancel"); self.cancel_button.setMinimumHeight(35)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_classification)
        button_layout.addWidget(self.classify_button); button_layout.addWidget(self.cancel_button)
        action_layout.addLayout(button_layout)
        main_layout.addWidget(action_group)

        results_group = QFrame(); results_group.setFrameShape(QFrame.Shape.StyledPanel)
        results_layout = QVBoxLayout(results_group); results_layout.addWidget(QLabel("<b>4. Results</b>"))
        self.results_list = QListWidget(); self.results_list.setMinimumHeight(100)
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
            screen = self.screen()
            if screen:
                screen_geometry = screen.availableGeometry()
                x = (screen_geometry.width() - self.width()) // 2
                y = (screen_geometry.height() - self.height()) // 2
                self.move(x, y)
        except Exception as e: print(f"Could not center window: {e}")

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
        if self.classification_thread and self.classification_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Classification is already in progress.")
            return

        model_path = self.model_file_widget.get_path()
        label_path = self.label_file_widget.get_path()
        input_folder = self.input_folder_widget.get_paths()
        output_folder = self.output_folder_widget.get_paths()

        if not all([model_path, label_path, input_folder, output_folder]):
            QMessageBox.warning(self, "Input Error", "All fields (Model, Labels, Input, Output) are required.")
            return

        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp"]
        all_image_files = []
        for ext in image_extensions:
            all_image_files.extend(glob.glob(os.path.join(input_folder, ext)))

        if not all_image_files:
            QMessageBox.information(self, "No Images", f"No supported image files found in {input_folder}.")
            self.progress_bar.reset(); self.progress_bar.setFormat("No images found")
            return

        image_files_to_process = sorted(all_image_files)[:self.MAX_IMAGE_FILES]
        if len(all_image_files) > self.MAX_IMAGE_FILES:
            QMessageBox.information(self, "Image Limit", f"Found {len(all_image_files)} images. Processing the first {self.MAX_IMAGE_FILES} (alphabetically).")

        self.save_settings()
        self.clear_results()
        self.statusBar().showMessage("Starting classification...")
        self.was_cancelled_or_error_during_processing = False

        self.classification_thread = ClassificationThread(model_path, label_path, image_files_to_process, output_folder, 16)
        self.classification_thread.progress_update.connect(self.update_progress)
        self.classification_thread.result_update.connect(self.add_result)
        self.classification_thread.finished.connect(self.classification_finished)
        self.classification_thread.error.connect(self.show_error)

        self.classify_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setFormat("%v/%m (%p%)")
        self.progress_bar.setMaximum(len(image_files_to_process))
        self.progress_bar.setValue(0)

        self.classification_thread.start()

    def cancel_classification(self):
        if self.classification_thread and self.classification_thread.isRunning():
            self.statusBar().showMessage("Canceling classification...")
            self.was_cancelled_or_error_during_processing = True
            self.classification_thread.stop()

    def update_progress(self, value, maximum):
        if maximum > 0:
            self.progress_bar.setMaximum(maximum)
            self.progress_bar.setValue(value)
            self.statusBar().showMessage(f"Processing: {value}/{maximum}")
        else:
            self.progress_bar.setMaximum(1)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("Processing...")


    def add_result(self, file_path, class_name, confidence):
        item_text = f"{os.path.basename(file_path)} → {class_name} (Conf: {confidence})"
        self.results_list.addItem(item_text)
        self.results_list.scrollToBottom()

    def classification_finished(self):
        current_status = self.statusBar().currentMessage()

        if self.was_cancelled_or_error_during_processing:
            if not current_status.startswith("Error:"):
                 self.statusBar().showMessage("Classification canceled or an error occurred.")
        else:
            if self.progress_bar.value() == 0 and self.progress_bar.maximum() > 0:
                self.statusBar().showMessage("Classification complete. No items were processed (check model/labels or for early error).")
            elif self.progress_bar.maximum() == 0 :
                self.statusBar().showMessage("Ready. No image files were processed.")
            else:
                self.statusBar().showMessage("Classification complete.")

        self.classify_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.classification_thread = None

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.statusBar().showMessage(f"Error: {error_message}")
        self.was_cancelled_or_error_during_processing = True

    def clear_results(self):
        self.results_list.clear()
        self.progress_bar.reset()
        self.progress_bar.setFormat("%v/%m (%p%)")
        if not (self.classification_thread and self.classification_thread.isRunning()):
            self.statusBar().showMessage("Results cleared. Ready.")

    def closeEvent(self, event):
        self.save_settings()
        thread_ref = self.classification_thread
        if thread_ref and thread_ref.isRunning():
            self.statusBar().showMessage("Closing: Stopping active classification...")
            self.was_cancelled_or_error_during_processing = True
            thread_ref.stop()
            if not thread_ref.wait(3000):
                QMessageBox.warning(self, "Closing", "Classification thread did not stop cleanly. Forcing exit.")
        event.accept()

def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    main_window = OpenMCX()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
