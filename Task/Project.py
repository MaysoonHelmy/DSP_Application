import tkinter as tk
from tkinter import ttk
import numpy as np
import pywt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

class EOGTaskPage(tk.Frame):
    def __init__(self, parent, task_name):
        super().__init__(parent, bg="#FFFFFF")

        # Initialize signal variables
        self.data = {}
        self.raw_data = {}
        self.scaler = StandardScaler()

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

        # Button frame for KNN button
        button_frame = tk.Frame(top_frame, bg="#FFFFFF")
        button_frame.pack(pady=10)

        knn_button = ttk.Button(button_frame, text="Train & Test KNN Model", command=self.knn_model)
        knn_button.grid(row=0, column=0, padx=5, pady=5)

        # Result text box
        self.result_text = tk.Text(top_frame, height=8, width=80, wrap=tk.WORD, font=("Helvetica", 12))
        self.result_text.pack(pady=10, fill=tk.X)

        # Create figure with two subplots vertically stacked
        self.figure = Figure(figsize=(10, 8))  # Adjust figure size for vertical layout
        self.plot_top = self.figure.add_subplot(211)  # Top plot for raw signal
        self.plot_bottom = self.figure.add_subplot(212)  # Bottom plot for processed signal
        self.canvas = FigureCanvasTkAgg(self.figure, bottom_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

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
            return signal if signal.size else np.array([])  # Ensure non-empty array
        except Exception:
            return np.array([])

    def load_data_from_file(self):
        """Load training and testing data from files"""
        try:
            self.raw_data['train_left'] = self.read_signal_from_file("right&left/train_left.txt")
            self.raw_data['test_left'] = self.read_signal_from_file("right&left/test_left.txt")
            self.raw_data['train_right'] = self.read_signal_from_file("right&left/train_right.txt")
            self.raw_data['test_right'] = self.read_signal_from_file("right&left/test_right.txt")

            # Also assign raw data to the data dictionary
            self.data = self.raw_data.copy()

        except Exception:
            raise

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        """Generate a bandpass filter"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        return butter(order, [low, high], btype='band')

    def apply_bandpass_filter(self, signal, b, a):
        """Apply bandpass filter to the signal"""
        if signal.size == 0:
            return np.array([])

        try:
            return filtfilt(b, a, signal.flatten())
        except Exception:
            return signal

    def normalize_signal(self, signal):
        """Normalize signal to [0, 1] range"""
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val == min_val:
            return np.zeros_like(signal)
        return (signal - min_val) / (max_val - min_val)

    def preprocess_data(self):
        """Preprocess data: remove DC component, bandpass filter, and normalize."""
        try:
            # Remove DC component (mean of the signal)
            for key in self.data:
                signal = self.data[key]
                if signal.size > 0:
                    # Remove the mean to eliminate DC component
                    self.data[key] = signal - np.mean(signal)

            # Bandpass filter setup
            lowcut = 0.5
            highcut = 50.0
            fs = 1000
            b, a = self.butter_bandpass(lowcut, highcut, fs)

            # Apply the bandpass filter and normalize the signal
            for key in self.data:
                signal = self.data[key]
                if signal.size > 0:
                    filtered_signal = self.apply_bandpass_filter(signal, b, a)
                    self.data[key] = self.normalize_signal(filtered_signal)

        except Exception as e:
            print(f"Error during preprocessing: {e}")

    def wavelet_feature_engineering(self, signal, wavelet='db4', level=4):
        """Perform wavelet feature extraction with enhanced depth and additional features."""
        if signal.size == 0:
            return np.zeros(5 * (level + 1))

        try:
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            features = []
            for coeff in coeffs:
                features.extend([
                    np.mean(coeff),            # Mean
                    np.std(coeff),             # Standard deviation
                    np.var(coeff),             # Variance
                    np.sum(np.square(coeff)),  # Energy (sum of squares)
                    np.max(coeff) - np.min(coeff)  # Range
                ])
            return np.array(features)
        except Exception as e:
            print(f"Error during wavelet feature extraction: {e}")
            return np.zeros(5 * (level + 1))

    def extract_features(self, signal, window_size=100, overlap=50):
        """Extract features from the signal using overlapping windows"""
        n_samples = len(signal)
        features = []

        for start in range(0, n_samples - window_size, window_size - overlap):
            end = start + window_size
            segment = signal[start:end]

            # Extract features for each segment
            feature_vector = self.wavelet_feature_engineering(segment)
            features.append(feature_vector)

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

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def run_model(self, model_name):
        """Train and test the model with improved strategies."""
        try:
            self.load_data_from_file()
            self.preprocess_data()

            # Extract features for each signal using overlapping windows
            train_left_features = self.extract_features(self.data['train_left'])
            train_right_features = self.extract_features(self.data['train_right'])
            test_left_features = self.extract_features(self.data['test_left'])
            test_right_features = self.extract_features(self.data['test_right'])

            # Combine features and create labels
            X_train = np.vstack([train_left_features, train_right_features])
            X_test = np.vstack([test_left_features, test_right_features])

            # Create labels for the samples: 0 for train_left and 1 for train_right
            y_train = np.array([0] * len(train_left_features) + [1] * len(train_right_features))
            y_test = np.array([0] * len(test_left_features) + [1] * len(test_right_features))

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Use StratifiedKFold for cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            base_model = KNeighborsClassifier()
            param_grid = {
                'n_neighbors': list(range(3, 15)),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }

            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=skf,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)

            # Train the best model
            best_model = grid_search.best_estimator_
            best_model.fit(X_train_scaled, y_train)

            # Test the model
            predicted = best_model.predict(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, predicted)
            report = classification_report(y_test, predicted, zero_division=0)

            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"{model_name} Model Results\n")
            self.result_text.insert(tk.END, f"Test Accuracy: {accuracy * 100:.2f}%\n")
            self.result_text.insert(tk.END, f"Classification Report:\n{report}")

            # Plot signals
            self.plot_signals()

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")

    def plot_signals(self):
        """Plot the left and right signals before and after preprocessing"""
        self.plot_top.clear()
        self.plot_bottom.clear()

        # Plot the left signal
        self.plot_top.plot(self.raw_data['train_left'], color='blue', label='Raw Left Signal')
        self.plot_bottom.plot(self.data['train_left'], color='red', label='Processed Left Signal')

        # Plot the right signal
        self.plot_top.plot(self.raw_data['train_right'], color='green', label='Raw Right Signal')
        self.plot_bottom.plot(self.data['train_right'], color='orange', label='Processed Right Signal')

        self.plot_top.set_title("Raw Signals")
        self.plot_bottom.set_title("Processed Signals")

        # Add labels and legend
        self.plot_top.set_xlabel("Samples")
        self.plot_top.set_ylabel("Amplitude")
        self.plot_bottom.set_xlabel("Samples")
        self.plot_bottom.set_ylabel("Amplitude")
        self.plot_top.legend()
        self.plot_bottom.legend()

        self.canvas.draw()

    def knn_model(self):
        """Train and test KNN model"""
        self.run_model("KNN")
