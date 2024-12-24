import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from task_page2 import SignalSamplesAreEqual

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

        correlate_button = ttk.Button(right_frame, text="Compute Correlation", command=self.compute_cross_correlation)
        correlate_button.pack(pady=10)

        self.correlation_plot_area = tk.Frame(right_frame, bg="#FFFFFF")
        self.correlation_plot_area.pack(fill="both", expand=True)

        remove_dc_component_label = tk.Label(right_frame, text="Remove the DC component", font=("Helvetica", 20, "bold"), bg="#FFFFFF")
        remove_dc_component_label.pack(pady=10)

        remove_dc_component_button = ttk.Button(right_frame, text="Compute the DC component", command=self.remove_dc_time_domain)
        remove_dc_component_button.pack(pady=10)

        self.remove_dc_component_plot_area = tk.Frame(right_frame, bg="#FFFFFF")
        self.remove_dc_component_plot_area.pack(fill="both", expand=True)

        self.signals = []

    def load_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Signal Files",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not file_paths:
            return None

        self.signals = []

        try:
            for file_path in file_paths:
                indices = []
                samples = []
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        values = line.strip().split()
                        if len(values) == 2:
                            indices.append(int(values[0]))
                            samples.append(float(values[1]))
                self.signals.append({"indices": indices, "samples": samples})

            messagebox.showinfo("Success", f"{len(self.signals)} signals loaded successfully.")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading files: {e}")
            return None

    def clear_plot_area(self, plot_area):
        for widget in plot_area.winfo_children():
            widget.destroy()

    def plot_result(self, data, title, plot_area):
        self.clear_plot_area(plot_area)

        figure = plt.Figure(figsize=(5, 3), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot(data, marker=" o")
        ax.set_title(title)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.grid(True)

        canvas = FigureCanvasTkAgg(figure, plot_area)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def plot_signal(self, index, signal, title, plot_area):
        self.clear_plot_area(plot_area)
        figure = plt.Figure(figsize=(5, 3), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot(index, signal)
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

        signal1 = self.signals[0]
        signal2 = self.signals[1]
        indices1, samples1 = signal1["indices"], signal1["samples"]
        indices2, samples2 = signal2["indices"], signal2["samples"]
        min_index = indices1[0] + indices2[0]
        max_index = indices1[-1] + indices2[-1]
        output_indices = list(range(min_index, max_index + 1))
        output_samples = []

        for n in output_indices:
            sum_value = 0
            for k in range(len(samples1)):
                x_index = indices1[k]
                h_index = n - x_index

                if h_index in indices2:
                    h_value = samples2[indices2.index(h_index)]
                else:
                    h_value = 0

                sum_value += samples1[k] * h_value
            output_samples.append(sum_value)
        self.ConvTest(output_indices, output_samples)
        self.plot_signal(output_indices, output_samples, "Convolution Signal", self.convolution_plot_area)

    def remove_dc_time_domain(self, signal_data):
        # Check if signal_data is empty
        if len(signal_data) < 1:
            messagebox.showerror("Error", "At least one signal is required for DC removal.")
            return

        # Assuming signal_data is a numpy array, compute the DC component
        dc_component = np.mean(signal_data)
        signal_no_dc = signal_data - dc_component

        # Round the signal values
        signal_no_dc_rounded = [round(float(val), 3) for val in signal_no_dc]

        # Print or save the result if needed
        print(signal_no_dc_rounded)
        return signal_no_dc_rounded

    def dft(self, signal):
        N = len(signal)
        dft_result = []
        for k in range(N):
            sum_value = sum(signal[n] * np.exp(-2j * np.pi * k * n / N) for n in range(N))
            dft_result.append(sum_value)
        return np.array(dft_result)

    def idft(self, dft_signal):
        N = len(dft_signal)
        idft_result = []
        for n in range(N):
            sum_value = sum(dft_signal[k] * np.exp(2j * np.pi * k * n / N) for k in range(N))
            idft_result.append(sum_value / N)
        return np.array(idft_result)

    def remove_dc_frequency_domain(self):
        signal1 = self.signals[0]
        signal = signal1["samples"]
        
        dft_signal = self.dft(signal)
        
        dft_signal[0] = 0
        
        signal_no_dc = self.idft(dft_signal)
        
        return np.real(signal_no_dc)
   

    def compute_cross_correlation(self):
       if not self.signals or len(self.signals) < 2:
           messagebox.showerror("Error", "Please load at least two signals.")
           return

       try:
           # Extract the signals
           signal1 = np.array(self.signals[0]["samples"][:5])
           signal2 = np.array(self.signals[1]["samples"][:5])

           len_signal1 = len(signal1)
           len_signal2 = len(signal2)

           if len_signal1 != len_signal2:
               messagebox.showerror("Error", "Signals must have the same length for correlation.")
               return

           print(f"Signal 1 Length: {len_signal1}")
           print(f"Signal 2 Length: {len_signal2}")

           # Pre-compute squared sums for normalization
           X1_squared_sum = np.sum(signal1 ** 2)
           X2_squared_sum = np.sum(signal2 ** 2)
           normalization = np.sqrt(X1_squared_sum * X2_squared_sum)

           # Compute the cross-correlation with periodic signals
           r12 = []
           for j in range(len_signal1):
               numerator = sum(signal1[i] * signal2[(i + j) % len_signal1] for i in range(len_signal1))  # Periodic signals
               r12.append(numerator / normalization)

           # Plot the results
           lags = list(range(len_signal1))
           self.plot_signal(lags, r12, "Cross-Correlation (Normalized)", self.correlation_plot_area)
           self.Compare_Signals("CorrOutput.txt", lags, r12)
           print("Computed Cross-Correlation Values (Normalized):", r12)

       except ValueError as e:
           messagebox.showerror("Error", f"Invalid input: {e}")
       except Exception as e:
           messagebox.showerror("Error", f"Error during cross-correlation: {e}")

    def compute_moving_average(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal file loaded.")
            return

        try:
            window_size = int(self.window_size_entry.get())
            if window_size <= 0:
                raise ValueError("Window size must be greater than 0.")

            # Using the first loaded signal for computation
            signal = self.signals[0]["samples"]
            indices = self.signals[0]["indices"]

            if len(signal) < window_size:
                messagebox.showerror("Error", "Window size cannot be larger than the signal length.")
                return

            # Compute the moving average
            smoothed_signal = [
                np.mean(signal[i:i + window_size])
                for i in range(len(signal) - window_size + 1)
            ]
            smoothed_indices = indices[:len(smoothed_signal)]

            # Perform signal comparison based on the window size
            if window_size == 3:
                SignalSamplesAreEqual("OutMovAvgTest1.txt", smoothed_indices, smoothed_signal)
            elif window_size == 5:
                SignalSamplesAreEqual("OutMovAvgTest2.txt", smoothed_indices, smoothed_signal)

            # Plot the smoothed signal
            self.plot_signal(smoothed_indices, smoothed_signal, f"Moving Average (Window Size: {window_size})", self.moving_avg_plot_area)

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def Compare_Signals(self,file_name,Your_indices,Your_samples):
        expected_indices=[]
        expected_samples=[]
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # process line
                L=line.strip()
                if len(L.split(' '))==2:
                    L=line.split(' ')
                    V1=int(L[0])
                    V2=float(L[1])
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                    line = f.readline()
                else:
                    break
        print("Current Output Test file is: ")
        print(file_name)
        print("\n")
        if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
            messagebox.showerror("Failed","Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(Your_indices)):
            if(Your_indices[i]!=expected_indices[i]):
                messagebox.showerror("Failed","Test case failed, your signal have different indicies from the expected one")
                return
        for i in range(len(expected_samples)):
            if abs(Your_samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                messagebox.showerror("Failed","Test case failed, your signal have different values from the expected one")
                return
        messagebox.showinfo("Sucess","Test case passed successfully")

    def ConvTest(self,Your_indices,Your_samples):
        """
            Test inputs
            InputIndicesSignal1 =[-2, -1, 0, 1]
            InputSamplesSignal1 = [1, 2, 1, 1 ]

            InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
            InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
        """

        expected_indices=[-2, -1, 0, 1, 2, 3, 4, 5, 6]
        expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1 ]


        if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
            messagebox.showerror("Failed","Conv Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(Your_indices)):
            if(Your_indices[i]!=expected_indices[i]):
                messagebox.showerror("Failed","Conv Test case failed, your signal have different indicies from the expected one")
                return
        for i in range(len(expected_samples)):
            if abs(Your_samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                messagebox.showerror("Failed","Conv Test case failed, your signal have different values from the expected one")
                return
        messagebox.showinfo("Sucess","Conv Test case passed successfully")

