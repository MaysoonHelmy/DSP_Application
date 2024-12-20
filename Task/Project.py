import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.signal import butter, filtfilt
from pywt import wavedec
import matplotlib.pyplot as plt
import seaborn as sns
from task_page6 import TaskPage6
from task_page7 import Task7
from sklearn.model_selection import GridSearchCV

class EOGTaskPage(tk.Frame):
    def __init__(self, parent, task_name):
        super().__init__(parent, bg="#FFFFFF")

        self.indices_array1 = None
        self.samples_array1 = None
        self.indices_array = None
        self.samples_array = None
        self.filtered_signal = None
        self.b = None
        self.a = None

        label = tk.Label(self, text=task_name, font=("Helvetica", 30, "bold"), bg="#FFFFFF")
        label.pack(pady=20)

        model_button = ttk.Button(self, text="Train & Test KNN Model", command=self.knn_model)
        model_button.pack(pady=10)

        self.data = {}

        self.result_label = tk.Label(self, text="Accuracy: ", font=("Helvetica", 14), bg="#FFFFFF")
        self.result_label.pack(pady=20)

    def normalize_signal(self, signal, method="0 to 1"):
        signal = signal.flatten()
        if method == "0 to 1":
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val == min_val:
                return np.zeros_like(signal)
            normalized = (signal - min_val) / (max_val - min_val)
        else:  # -1 to 1
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val == min_val:
                return np.zeros_like(signal)
            normalized = 2 * ((signal - min_val) / (max_val - min_val)) - 1

        return normalized

    def create_windows(self, signal, window_size=200, stride=100):
        windows = []
        for i in range(0, len(signal) - window_size + 1, stride):
            window = signal[i:i + window_size]
            windows.append(window)
        return np.array(windows)

    def downsample_signal(self, indices, samples, factor):
        try:
            downsampled = Task7.downsample(self, indices, samples, factor)
            return np.array(downsampled)
        except Exception as e:
            print(f"Error in downsample_signal: {e}")
            return np.array([])

    def read_signal_from_file(self, file_path):
        try:
            signal = []
            with open(file_path, 'r') as file:
                for line in file:
                    values = [float(val) for val in line.strip().split()]
                    if values:
                        signal.extend(values)
            signal = np.array(signal)
            if signal.size == 0:
                return np.array([[]])
            return signal.reshape(-1, 1)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return np.array([[]])

    def load_data_from_file(self):
        try:
            train_left_file = "right&left/train_left.txt"
            test_left_file = "right&left/test_left.txt"
            train_right_file = "right&left/train_right.txt"
            test_right_file = "right&left/test_right.txt"

            self.data['train_left'] = self.read_signal_from_file(train_left_file)
            self.data['test_left'] = self.read_signal_from_file(test_left_file)
            self.data['train_right'] = self.read_signal_from_file(train_right_file)
            self.data['test_right'] = self.read_signal_from_file(test_right_file)

        except Exception as e:
            print(f"Error in load_data_from_file: {e}")
            raise

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        return butter(order, [low, high], btype='band')

    def apply_bandpass_filter(self, signal):
        if signal.size == 0:
            return np.array([])

        signal = signal.flatten()

        try:
            filtered_signal = filtfilt(self.b, self.a, signal)
            return filtered_signal
        except Exception as e:
            print(f"Error in filtering: {e}")
            return signal

    def preprocess_data(self):
        try:
            lowcut = 0.5
            highcut = 50.0
            fs = 1000

            self.b, self.a = self.butter_bandpass(lowcut, highcut, fs)
            target_length = 1000  # Fixed length for all signals

            for signal_type in ['train_left', 'train_right', 'test_left', 'test_right']:
                if signal_type in self.data and self.data[signal_type].size > 0:
                    signal = np.array(self.data[signal_type])

                    # Apply bandpass filter
                    filtered_signal = self.apply_bandpass_filter(signal)

                    # Normalize
                    normalized_signal = self.normalize_signal(filtered_signal, "0 to 1")

                    # Remove DC component
                    dc_removed = np.array(TaskPage6.remove_dc_time_domain(self, normalized_signal))

                    # Downsample
                    downsampled = self.downsample_signal(
                        np.arange(len(dc_removed)),
                        dc_removed,
                        88
                    )

                    # Ensure consistent length
                    if len(downsampled) < target_length:
                        self.data[signal_type] = np.pad(
                            downsampled,
                            (0, target_length - len(downsampled))
                        )
                    else:
                        self.data[signal_type] = downsampled[:target_length]


        except Exception as e:
            print(f"Error in preprocess_data: {e}")
            raise

    def wavelet_feature_engineering(self, signal_data):
        if not isinstance(signal_data, np.ndarray):
            signal_data = np.array(signal_data)

        if signal_data.size == 0:
            return np.array([])

        signal_data = signal_data.flatten()

        try:
            # Perform Discrete Wavelet Transform (DWT)
            coeffs = wavedec(signal_data, 'db4', level=5)
            features = []
            for coeff in coeffs:
                features.extend([np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff)])
            return np.array(features)
        except Exception as e:
            print(f"Error in wavelet_feature_engineering: {e}")
            return np.array([])

    def extract_features_from_windows(self, signal, window_size=200, stride=100):
        windows = self.create_windows(signal, window_size, stride)
        features = []
        for window in windows:
            window_features = self.wavelet_feature_engineering(window)
            features.append(window_features)
        return np.array(features)

    def knn_model(self):
        try:
            self.load_data_from_file()
            self.preprocess_data()

            # Extract features using sliding windows for preprocessed training data
            window_size = 200
            stride = 100

            # Process training data after preprocessing
            X_train_left = self.extract_features_from_windows(self.data['train_left'], window_size, stride)
            X_train_right = self.extract_features_from_windows(self.data['train_right'], window_size, stride)

            # Process test data
            X_test_left = self.extract_features_from_windows(self.data['test_left'], window_size, stride)
            X_test_right = self.extract_features_from_windows(self.data['test_right'], window_size, stride)

            # Combine features
            X_train = np.vstack((X_train_left, X_train_right))
            X_test = np.vstack((X_test_left, X_test_right))

            # Create labels
            y_train = np.array([0] * len(X_train_left) + [1] * len(X_train_right), dtype=int)
            y_test = np.array([0] * len(X_test_left) + [1] * len(X_test_right), dtype=int)

            # Grid search for hyperparameter tuning
            param_grid = {'n_neighbors': range(1, 11), 'weights': ['uniform', 'distance']}
            grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            predictions = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            self.result_label.config(text=f"Accuracy: {accuracy * 100:.2f}%")
            self.plot_model_function(grid_search.best_params_, accuracy)
            self.plot_confusion_matrix(y_test, predictions)

        except Exception as e:
            print(f"Error in KNN model: {e}")
            self.result_label.config(text="Error in processing. Check console for details.")

    def plot_model_function(self, best_params, accuracy):
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            ax.plot(best_params['n_neighbors'], accuracy, 'ro')

            ax.set_xlabel('Number of Neighbors')
            ax.set_ylabel('Accuracy')
            ax.set_title('KNN Hyperparameter Tuning')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in plot_model_function: {e}")

    def plot_confusion_matrix(self, y_test, predictions):
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Left", "Right"], yticklabels=["Left", "Right"])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
