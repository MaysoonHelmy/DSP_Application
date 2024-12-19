import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.signal import butter, filtfilt
from pywt import wavedec
import matplotlib.pyplot as plt
from task_page6 import TaskPage6
from task_page7 import Task7
from task_page2 import TaskPage2

class EOGTaskPage(tk.Frame):
    def __init__(self, parent, task_name):
        super().__init__(parent, bg="#FFFFFF")

        self.indices_array1 = None
        self.samples_array1 = None
        self.indices_array = None
        self.samples_array = None
        self.filtered_signal = None

        label = tk.Label(self, text=task_name, font=("Helvetica", 30, "bold"), bg="#FFFFFF")
        label.pack(pady=20)

        # Remove the load button as no files will be loaded
        model_button = ttk.Button(self, text="Train & Test KNN Model", command=self.knn_model)
        model_button.pack(pady=10)

        self.data = {}

        # Add a result label to display the accuracy
        self.result_label = tk.Label(self, text="Accuracy: ", font=("Helvetica", 14), bg="#FFFFFF")
        self.result_label.pack(pady=20)

    def load_data_from_file(self):
        # Specify the file paths for the training and testing data
        train_left_file = "right&left/train_left.txt"
        test_left_file = "right&left/test_left.txt"
        train_right_file = "right&left/train_right.txt"
        test_right_file = "right&left/test_right.txt"

        # Read the files for left and right signals using the paths passed manually
        self.data['train_left'] = self.read_signal_from_file(train_left_file)
        self.data['test_left'] = self.read_signal_from_file(test_left_file)
        self.data['train_right'] = self.read_signal_from_file(train_right_file)
        self.data['test_right'] = self.read_signal_from_file(test_right_file)

        # Manually defining labels (as an example, adjust according to your dataset)
        # These should be loaded from a file or defined properly for classification
        self.data['train_labels'] = np.zeros(len(self.data['train_left']))  # Dummy labels for left data
        self.data['test_labels'] = np.zeros(len(self.data['test_left']))  # Dummy labels for test data
        # You can add right labels similarly or define how they are calculated

    def read_signal_from_file(self, file_path):
        # Read the signal data from a text file
        signal = []
        with open(file_path, 'r') as file:
            for line in file:
                # Split values by space or comma and convert to float
                values = line.split()
                signal.extend([float(value) for value in values])
        return np.array(signal)

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(self, signal, lowcut=0.5, highcut=20, fs=176, order=4):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order)
        # Ensure signal is 1D before applying the filter
        if signal.ndim > 1:
            signal = signal.flatten()  # Flatten to 1D if necessary
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal


    def preprocess_data(self):
        # Remove DC component before Bandpass Filtering for left and right signals
        self.data['train_left'] = TaskPage6.remove_dc_time_domain(self, self.data['train_left'])
        self.data['test_left'] = TaskPage6.remove_dc_time_domain(self, self.data['test_left'])
        self.data['train_right'] = TaskPage6.remove_dc_time_domain(self, self.data['train_right'])
        self.data['test_right'] = TaskPage6.remove_dc_time_domain(self, self.data['test_right'])

        # Apply Bandpass Filter after DC removal for left and right signals
        self.data['train_left'] = np.array([self.apply_bandpass_filter(signal) for signal in self.data['train_left']])
        self.data['test_left'] = np.array([self.apply_bandpass_filter(signal) for signal in self.data['test_left']])
        self.data['train_right'] = np.array([self.apply_bandpass_filter(signal) for signal in self.data['train_right']])
        self.data['test_right'] = np.array([self.apply_bandpass_filter(signal) for signal in self.data['test_right']])

        # Normalize the data for left and right signals
        self.data['train_left'] = TaskPage2.normalize_signal(self.data['train_left'], "0 to 1")
        self.data['test_left'] = TaskPage2.normalize_signal(self.data['test_left'], "0 to 1")
        self.data['train_right'] = TaskPage2.normalize_signal(self.data['train_right'], "0 to 1")
        self.data['test_right'] = TaskPage2.normalize_signal(self.data['test_right'], "0 to 1")

        # Get the indices and samples before downsampling
        self.indices_array = np.arange(len(self.data['train_left']))
        self.samples_array = self.data['train_left']

        self.indices_array1 = np.arange(len(self.data['test_left']))
        self.samples_array1 = self.data['test_left']

        # Downsample the data for left and right signals
        self.data['train_left'] = Task7.downsample(self, self.indices_array, self.samples_array, 88)
        self.data['test_left'] = Task7.downsample(self, self.indices_array1, self.samples_array1, 88)

        self.indices_array2 = np.arange(len(self.data['train_right']))
        self.samples_array2 = self.data['train_right']

        self.indices_array3 = np.arange(len(self.data['test_right']))
        self.samples_array3 = self.data['test_right']

        # Downsample the data for right signals
        self.data['train_right'] = Task7.downsample(self, self.indices_array2, self.samples_array2, 88)
        self.data['test_right'] = Task7.downsample(self, self.indices_array3, self.samples_array3, 88)

    def feature_engineering(self, signal_data):
        # Apply wavelet transform
        coeffs = wavedec(signal_data, 'db4', level=4)

        # Flatten the coefficients to use them as features for machine learning
        features = np.hstack([coeff.flatten() for coeff in coeffs])

        return features

    def knn_model(self):
        # Load data directly
        self.load_data_from_file()

        # Apply Preprocessing
        self.preprocess_data()

        # Apply Feature engineering for left and right signals
        X_train_left = np.array([self.feature_engineering(signal) for signal in self.data['train_left']])
        X_test_left = np.array([self.feature_engineering(signal) for signal in self.data['test_left']])

        X_train_right = np.array([self.feature_engineering(signal) for signal in self.data['train_right']])
        X_test_right = np.array([self.feature_engineering(signal) for signal in self.data['test_right']])

        # Concatenate left and right signals' features for training
        X_train = np.vstack((X_train_left, X_train_right))
        X_test = np.vstack((X_test_left, X_test_right))

        # Concatenate left and right labels for training
        y_train = np.hstack((self.data['train_labels'], self.data['train_labels']))
        y_test = np.hstack((self.data['test_labels'], self.data['test_labels']))

        # Train the KNN model
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        # Predict the test labels
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Update the result_label to show the accuracy in the GUI
        self.result_label.config(text=f"Accuracy: {accuracy * 100:.2f}%")

        # Show the plot for model performance (Confusion Matrix or other)
        self.plot_model_performance(y_test, predictions)

    def plot_model_performance(self, y_true, y_pred):
        # Confusion Matrix Plot
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
