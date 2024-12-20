import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import butter, filtfilt
from pywt import wavedec
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from task_page6 import TaskPage6
from task_page7 import Task7

class EOGTaskPage(tk.Frame):
    def __init__(self, parent, task_name):
        super().__init__(parent, bg="#FFFFFF")

        # Initialize signal variables
        self.indices_array1 = None
        self.samples_array1 = None
        self.indices_array = None
        self.samples_array = None
        self.filtered_signal = None
        self.b = None
        self.a = None

        # Create main container
        main_container = tk.Frame(self, bg="#FFFFFF")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top frame for controls
        top_frame = tk.Frame(main_container, bg="#FFFFFF")
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Bottom frame for plots
        bottom_frame = tk.Frame(main_container, bg="#FFFFFF")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Add title
        label = tk.Label(top_frame, text=task_name, font=("Helvetica", 30, "bold"), bg="#FFFFFF")
        label.pack(pady=20)

        # Button frame for model buttons
        button_frame = tk.Frame(top_frame, bg="#FFFFFF")
        button_frame.pack(pady=10)

        knn_button = ttk.Button(button_frame, text="Train & Test KNN Model", command=self.knn_model)
        knn_button.grid(row=0, column=0, padx=5, pady=5)

        svm_button = ttk.Button(button_frame, text="Train & Test SVM Model", command=self.svm_model)
        svm_button.grid(row=0, column=1, padx=5, pady=5)

        tree_button = ttk.Button(button_frame, text="Train & Test Decision Tree", command=self.tree_model)
        tree_button.grid(row=0, column=2, padx=5, pady=5)

        # Result text box
        self.result_text = tk.Text(top_frame, height=8, width=80, wrap=tk.WORD, font=("Helvetica", 12))
        self.result_text.pack(pady=10, fill=tk.X)

        # Create figure with two subplots
        self.figure = Figure(figsize=(10, 4))  # Adjust figure size
        self.plot_left = self.figure.add_subplot(121)  # Left plot
        self.plot_right = self.figure.add_subplot(122)  # Right plot
        self.canvas = FigureCanvasTkAgg(self.figure, bottom_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.data = {}
        self.scaler = StandardScaler()

    def plot_comparison(self, raw_signal_left, processed_signal_left, raw_signal_right, processed_signal_right):
        """Plot both left and right signals side by side"""

        # Left plot
        self.plot_left.clear()
        self.plot_left.plot(raw_signal_left, 'r-', alpha=0.6)
        self.plot_left.plot(processed_signal_left, 'b-', alpha=0.6)
        self.plot_left.set_title("Left Signal")
        self.plot_left.set_xlabel("Samples")
        self.plot_left.get_yaxis().set_visible(False)  # Remove Y-axis labels
        self.plot_left.legend(loc="upper right")
        self.plot_left.grid(True)

        # Right plot
        self.plot_right.clear()
        self.plot_right.plot(raw_signal_right, 'r-', alpha=0.6)
        self.plot_right.plot(processed_signal_right, 'b-', alpha=0.6)
        self.plot_right.set_title("Right Signal")
        self.plot_right.set_xlabel("Samples")
        self.plot_right.get_yaxis().set_visible(False)  # Remove Y-axis labels
        self.plot_right.legend(loc="upper right")
        self.plot_right.grid(True)

        self.canvas.draw()

    def normalize_signal(self, signal, method="0 to 1"):
        """Normalize signal to either [0, 1] or [-1, 1] range"""
        signal = signal.flatten()
        if method == "0 to 1":
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val == min_val:
                return np.zeros_like(signal)
            return (signal - min_val) / (max_val - min_val)
        else:  # Normalize to [-1, 1]
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val == min_val:
                return np.zeros_like(signal)
            return 2 * ((signal - min_val) / (max_val - min_val)) - 1

    def create_windows(self, signal, window_size=200, stride=100):
        """Create sliding windows from the signal"""
        windows = []
        for i in range(0, len(signal) - window_size + 1, stride):
            window = signal[i:i + window_size]
            windows.append(window)
        return np.array(windows)

    def downsample_signal(self, indices, samples, factor):
        """Downsample the signal by a given factor"""
        try:
            downsampled = Task7.downsample(self, indices, samples, factor)
            return np.array(downsampled)
        except Exception:
            return np.array([])

    def read_signal_from_file(self, file_path):
        """Read signal data from a file"""
        try:
            signal = []
            with open(file_path, 'r') as file:
                for line in file:
                    values = [float(val) for val in line.strip().split()]
                    if values:
                        signal.extend(values)
            signal = np.array(signal)
            return signal.reshape(-1, 1) if signal.size else np.array([[]])
        except Exception:
            return np.array([[]])

    def load_data_from_file(self):
        """Load training and testing data from files"""
        try:
            train_left_file = "right&left/train_left.txt"
            test_left_file = "right&left/test_left.txt"
            train_right_file = "right&left/train_right.txt"
            test_right_file = "right&left/test_right.txt"

            self.data['train_left'] = self.read_signal_from_file(train_left_file)
            self.data['test_left'] = self.read_signal_from_file(test_left_file)
            self.data['train_right'] = self.read_signal_from_file(train_right_file)
            self.data['test_right'] = self.read_signal_from_file(test_right_file)

        except Exception:
            raise

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        """Generate a bandpass filter"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        return butter(order, [low, high], btype='band')

    def apply_bandpass_filter(self, signal):
        """Apply bandpass filter to the signal"""
        if signal.size == 0:
            return np.array([])

        signal = signal.flatten()
        try:
            filtered_signal = filtfilt(self.b, self.a, signal)
            return filtered_signal
        except Exception:
            return signal

    def preprocess_data(self):
        """Preprocess data: bandpass filter, normalize, downsample"""
        try:
            lowcut = 0.5
            highcut = 50.0
            fs = 1000
            self.b, self.a = self.butter_bandpass(lowcut, highcut, fs)

            target_length = 1000

            left_signal_raw = None
            right_signal_raw = None
            processed_left_signal = None
            processed_right_signal = None

            for signal_type in ['train_left', 'train_right']:
                if signal_type in self.data and self.data[signal_type].size > 0:
                    signal = np.array(self.data[signal_type])

                    # Store raw signal for plotting
                    raw_signal = signal.flatten()

                    # Apply preprocessing
                    filtered_signal = self.apply_bandpass_filter(signal)
                    normalized_signal = self.normalize_signal(filtered_signal, "0 to 1")
                    dc_removed = np.array(TaskPage6.remove_dc_time_domain(self, normalized_signal))
                    downsampled = self.downsample_signal(np.arange(len(dc_removed)), dc_removed, 88)

                    # Ensure consistent signal length
                    if len(downsampled) < target_length:
                        processed_signal = np.pad(downsampled, (0, target_length - len(downsampled)))
                    else:
                        processed_signal = downsampled[:target_length]

                    self.data[signal_type] = processed_signal

                    # Collect raw and processed signals for plotting
                    if signal_type == 'train_left':
                        left_signal_raw = raw_signal
                        processed_left_signal = processed_signal
                    elif signal_type == 'train_right':
                        right_signal_raw = raw_signal
                        processed_right_signal = processed_signal

            # Now plot both left and right signals side by side
            if left_signal_raw is not None and processed_left_signal is not None and right_signal_raw is not None and processed_right_signal is not None:
                self.plot_comparison(left_signal_raw, processed_left_signal, right_signal_raw, processed_right_signal)

        except Exception:
            raise

    def wavelet_feature_engineering(self, signal_data):
        """Perform wavelet feature extraction"""
        if not isinstance(signal_data, np.ndarray):
            signal_data = np.array(signal_data)

        if signal_data.size == 0:
            return np.zeros(20)  # Return 20 zeros (5 stats × 4 levels)

        signal_data = signal_data.flatten()

        try:
            coeffs = wavedec(signal_data, 'db4', level=4)
            features = []
            for coeff in coeffs:
                features.extend([np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff)])
            return np.array(features)
        except Exception:
            return np.zeros(20)  # Return 20 zeros (5 stats × 4 levels)

    def extract_features_from_windows(self, signal, window_size=200, stride=100):
        """Extract features from signal windows"""
        windows = self.create_windows(signal, window_size, stride)
        features = []
        for window in windows:
            window_features = self.wavelet_feature_engineering(window)
            features.append(window_features)
        return np.array(features)

    def get_best_parameters(self, model_type, X, y):
        """Find best parameters using GridSearchCV"""
        if model_type == "KNN":
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            base_model = KNeighborsClassifier()

        elif model_type == "SVM":
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
            base_model = SVC()

        else:  # Decision Tree
            param_grid = {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = DecisionTreeClassifier()

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def run_model(self, model, model_name):
        """Train and test the model with cross-validation"""
        try:
            self.load_data_from_file()
            self.preprocess_data()

            # Prepare features
            train_left_features = self.extract_features_from_windows(self.data['train_left'])
            train_right_features = self.extract_features_from_windows(self.data['train_right'])
            test_left_features = self.extract_features_from_windows(self.data['test_left'])
            test_right_features = self.extract_features_from_windows(self.data['test_right'])

            # Combine features and create labels
            X_train = np.vstack([train_left_features, train_right_features])
            X_test = np.vstack([test_left_features, test_right_features])

            y_train = np.array([0] * len(train_left_features) + [1] * len(train_right_features))
            y_test = np.array([0] * len(test_left_features) + [1] * len(test_right_features))

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Find best model parameters using cross-validation
            best_model = self.get_best_parameters(model_name, X_train_scaled, y_train)

            # Perform cross-validation
            cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)

            # Train the model with best parameters
            best_model.fit(X_train_scaled, y_train)

            # Test the model
            predicted = best_model.predict(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, predicted)
            report = classification_report(y_test, predicted, zero_division=0)

            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"{model_name} Model Results\n")
            self.result_text.insert(tk.END, f"Best Parameters: {best_model.get_params()}\n\n")
            self.result_text.insert(tk.END, f"Cross-validation scores: {cv_scores}\n")
            self.result_text.insert(tk.END, f"Mean CV accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})\n\n")
            self.result_text.insert(tk.END, f"Test Accuracy: {accuracy * 100:.2f}%\n")
            self.result_text.insert(tk.END, f"Classification Report:\n{report}")

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")

    def knn_model(self):
        """Train and test KNN model"""
        self.run_model(KNeighborsClassifier(), "KNN")

    def svm_model(self):
        """Train and test SVM model"""
        self.run_model(SVC(), "SVM")

    def tree_model(self):
        """Train and test Decision Tree"""
        self.run_model(DecisionTreeClassifier(), "Decision Tree")