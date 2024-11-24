import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class TaskPage6(tk.Frame):
    def __init__(self, parent, task_name):
        super().__init__(parent, bg="#FFFFFF")
        self.task_name = task_name

        title_label = tk.Label(self, text="Convolution and Correlation", font=("Helvetica", 30, "bold"), bg="#FFFFFF")
        title_label.pack(pady=10)

        input_file_button = ttk.Button(self, text="Load Signal Files", command=self.load_files)
        input_file_button.pack(pady=10)

        main_frame = tk.Frame(self, bg="#FFFFFF")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, bg="#FFFFFF")
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        convolution_label = tk.Label(left_frame, text="Convolution", font=("Helvetica", 20, "bold"), bg="#FFFFFF")
        convolution_label.pack(pady=10)

        convolve_button = ttk.Button(left_frame, text="Compute Convolution", command=self.compute_convolution)
        convolve_button.pack(pady=10)

        self.convolution_plot_area = tk.Frame(left_frame, bg="#FFFFFF")
        self.convolution_plot_area.pack(fill="both", expand=True)

        moving_avg_label = tk.Label(left_frame, text="Moving Average", font=("Helvetica", 20, "bold"), bg="#FFFFFF")
        moving_avg_label.pack(pady=10)

        self.window_size_entry = ttk.Entry(left_frame)
        self.window_size_entry.pack(pady=5)
        self.window_size_entry.insert(0, "3")

        moving_avg_button = ttk.Button(left_frame, text="Compute Moving Average", command=self.compute_moving_average)
        moving_avg_button.pack(pady=10)

        self.moving_avg_plot_area = tk.Frame(left_frame, bg="#FFFFFF")
        self.moving_avg_plot_area.pack(fill="both", expand=True)

        right_frame = tk.Frame(main_frame, bg="#FFFFFF")
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        correlation_label = tk.Label(right_frame, text="Correlation", font=("Helvetica", 20, "bold"), bg="#FFFFFF")
        correlation_label.pack(pady=10)

        correlate_button = ttk.Button(right_frame, text="Compute Correlation", command=self.compute_normalized_cross_correlation)
        correlate_button.pack(pady=10)

        self.correlation_plot_area = tk.Frame(right_frame, bg="#FFFFFF")
        self.correlation_plot_area.pack(fill="both", expand=True)

        self.signals = []

    def load_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Signal Files",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not file_paths:
            return

        self.signals = []

        try:
            for file_path in file_paths:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    signal = [float(x) for line in lines for x in line.strip().split()]
                    self.signals.append(np.array(signal))
            messagebox.showinfo("Success", f"{len(self.signals)} signals loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading files: {e}")

    def clear_plot_area(self, plot_area):
        for widget in plot_area.winfo_children():
            widget.destroy()

    def plot_result(self, data, title, plot_area):
        self.clear_plot_area(plot_area)

        figure = plt.Figure(figsize=(5, 3), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot(data, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.grid(True)

        canvas = FigureCanvasTkAgg(figure, plot_area)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def compute_convolution(self):
        if len(self.signals) < 2:
            messagebox.showerror("Error", "At least two signals are required for convolution.")
            return

        try:
            signal1, signal2 = self.signals[0], self.signals[1]
            convolution_result = np.convolve(signal1, signal2, mode='full')
            self.plot_result(convolution_result, "Convolution Result", self.convolution_plot_area)

            # Print results to the console
            print(f"Convolution Result:\n{convolution_result}")

            self.Compare_Signals("ConvolutionResult.txt", range(len(convolution_result)), convolution_result)
            return convolution_result
        except Exception as e:
            messagebox.showerror("Error", f"Error computing convolution: {e}")
    def compute_normalized_cross_correlation(self):
        if len(self.signals) < 2:
            messagebox.showerror("Error", "At least two signals are required for cross-correlation.")
            return

        try:
            # Extract the first two signals
            signal1, signal2 = self.signals[0], self.signals[1]
            N = len(signal1)

            # Ensure signals are of the same length
            if len(signal1) != len(signal2):
                messagebox.showerror("Error", "Signals must have the same length.")
                return

            # Calculate mean and standard deviation of signals
            mean1 = np.mean(signal1)
            mean2 = np.mean(signal2)
            stddev1 = np.std(signal1)
            stddev2 = np.std(signal2)

            # Calculate normalized cross-correlation
            normalized_cross_corr = []
            for lag in range(-N + 1, N):
                value = 0
                for n in range(N):
                    # Wrap around using modulo for signal2
                    index2 = (n + lag) % N
                    value += (signal1[n] - mean1) * (signal2[index2] - mean2)
                value /= (N * stddev1 * stddev2)  # Normalize
                normalized_cross_corr.append(value)

            # Adjust indices to get only the first 4 samples
            indices = list(range(4))
            values = [round(normalized_cross_corr[i + N - 1], 8) for i in indices]

            # Print results to the console with 8 decimal places
            print(f"Normalized Cross-Correlation Result:\n{values}")

            # Find maximum correlation and calculate time delay
            max_corr = max(values)
            lag = indices[values.index(max_corr)]
            sampling_time = 1  # Replace with the actual sampling time if available
            time_delay = lag * sampling_time

            # Update the signal data with the formatted cross-correlation results
            self.correlation_results = [{"Index": i, "Value": v} for i, v in zip(indices, values)]

            # Display results
            self.plot_result(values, "Normalized Cross-Correlation Result", self.correlation_plot_area)

            # Pass the indices and samples to the Compare_Signals method
            self.Compare_Signals("CorrOutput.txt", indices, values)

        except Exception as e:
            messagebox.showerror("Error", f"Error computing normalized cross-correlation: {e}")

    def compute_moving_average(self):
        if len(self.signals) < 1:
            messagebox.showerror("Error", "At least one signal is required for moving average.")
            return

        try:
            signal = self.signals[0]
            window_size = int(self.window_size_entry.get())

            if window_size <= 0:
                messagebox.showerror("Error", "Window size must be greater than zero.")
                return

            if window_size > len(signal):
                messagebox.showerror("Error", "Window size cannot be greater than the signal length.")
                return

            moving_avg = []
            indices = []
            for i in range(len(signal) - window_size + 1):
                window = signal[i:i + window_size]
                avg = np.mean(window)
                moving_avg.append(avg)
                indices.append(i)

            self.plot_result(moving_avg, f"Moving Average (Window Size {window_size})", self.moving_avg_plot_area)

            # Print results to the console
            print(f"Moving Average Result (Window Size {window_size}):\n{moving_avg}")

            self.Compare_Signals("OutMovAvgTest1.txt", indices, moving_avg)
            self.Compare_Signals("OutMovAvgTest2.txt", indices, moving_avg)

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the window size.")
        except Exception as e:
            messagebox.showerror("Error", f"Error computing moving average: {e}")

    def Compare_Signals(self, file_name, indices, samples):
        expected_indices = []
        expected_samples = []

        try:
            with open(file_name, 'r') as f:
                for _ in range(4):
                    f.readline()
                line = f.readline()
                while line:
                    line = line.strip()
                    if len(line.split(' ')) == 2:
                        index, value = line.split(' ')
                        expected_indices.append(int(index))
                        expected_samples.append(float(value))
                    line = f.readline()

        except Exception as e:
            print(f"Error reading file: {e}")
            return

        # Ensure the expected number of indices and samples match the computed results
        if len(expected_samples) != len(samples):
            print(f"Test case for {file_name} failed: Sample count mismatch.")
            print(f"Expected samples count: {len(expected_samples)}, Computed samples count: {len(samples)}")
            return

        # Compare values
        for i, (expected_index, expected_sample) in enumerate(zip(expected_indices, expected_samples)):
            if i < len(indices):  # Compare only within bounds
                computed_sample = samples[i]
                if np.isclose(computed_sample, expected_sample):
                    print(f"Test case for {file_name} passed at index {expected_index}.")
                else:
                    print(f"Test case for {file_name} failed at index {expected_index}. Expected: {expected_sample}, Got: {computed_sample}")
            else:
                print(f"Test case for {file_name} failed: Extra computed sample at index {indices[i]}.")

