import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TaskPage2(tk.Frame):
    def __init__(self, parent, task_name):
        super().__init__(parent, bg="#FFFFFF")
        self.task_name = task_name

        # Split the page into two frames: one for controls (left) and one for the plot (right)
        self.left_frame = tk.Frame(self, bg="#FFFFFF", width=300)
        self.left_frame.pack(side="left", fill="y", padx=20, pady=20)

        self.right_frame = tk.Frame(self, bg="#FFFFFF")
        self.right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        title_label = tk.Label(self.left_frame, text=task_name, font=("Helvetica", 24, "bold"), bg="#FFFFFF")
        title_label.pack(pady=20)

        load_button = ttk.Button(self.left_frame, text="Load Signals", command=self.load_signals, width=20)
        load_button.pack(pady=10)

        tk.Label(self.left_frame, text="Select Operation", font=("Helvetica", 15), bg="#FFFFFF").pack(pady=5)
        self.operation_var = tk.StringVar(value="Addition")
        operations = ["Addition", "Subtraction", "Multiplication", "Squaring", "Normalization", "Accumulation"]
        operation_menu = ttk.Combobox(self.left_frame, values=operations, textvariable=self.operation_var, state='readonly')
        operation_menu.pack(pady=10)

        self.const_label = tk.Label(self.left_frame, text="Enter Constant for Multiplication:", font=("Helvetica", 14), bg="#FFFFFF")
        self.const_entry = tk.Entry(self.left_frame, font=("Helvetica", 12))
        self.const_label.pack_forget()
        self.const_entry.pack_forget()

        operation_menu.bind("<<ComboboxSelected>>", self.show_constant_entry)

        self.norm_var = tk.StringVar(value="-1 to 1")
        self.norm_label = tk.Label(self.left_frame, text="Normalization Range:", font=("Helvetica", 14), bg="#FFFFFF")
        self.norm_range = ttk.Combobox(self.left_frame, values=["-1 to 1", "0 to 1"], textvariable=self.norm_var, state='readonly')
        self.norm_label.pack_forget()
        self.norm_range.pack_forget()

        apply_button = ttk.Button(self.left_frame, text="Apply Operation", command=self.apply_operation, width=20)
        apply_button.pack(pady=20)

        self.plot_frame = tk.Frame(self.right_frame, bg="#FFFFFF")
        self.plot_frame.pack(fill="both", expand=True)

        self.signals = []

    def load_signals(self):
        file_paths = filedialog.askopenfilenames(title="Open Signal Files", filetypes=[("Text Files", "*.txt")])
        if file_paths:
            self.signals = []
            try:
                for file_path in file_paths:
                    with open(file_path, 'r') as file:
                        lines = file.readlines()


                    signal_type = int(lines[0].strip())
                    if signal_type != 0:
                        continue

                    is_periodic = int(lines[1].strip())
                    num_samples = int(lines[2].strip())

                    signal_data = np.array([list(map(float, line.strip().split())) for line in lines[3:3 + num_samples]])
                    self.signals.append(signal_data)

                messagebox.showinfo("Success", f"Successfully loaded {len(file_paths)} signals.")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading signals: {e}")

    def show_constant_entry(self, event):
        operation = self.operation_var.get()
        if operation == "Multiplication":
            self.const_label.pack(pady=5)
            self.const_entry.pack(pady=5)
            self.norm_label.pack_forget()
            self.norm_range.pack_forget()
        elif operation == "Normalization":
            self.norm_label.pack(pady=5)
            self.norm_range.pack(pady=5)
            self.const_label.pack_forget()
            self.const_entry.pack_forget()
        else:
            self.const_label.pack_forget()
            self.const_entry.pack_forget()
            self.norm_label.pack_forget()
            self.norm_range.pack_forget()

    def apply_operation(self):
        operation = self.operation_var.get()
        if not self.signals:
            messagebox.showerror("Error", "Please load at least one signal.")
            return

        try:
            if operation == "Addition":
                result_signal = self.add_signals()
            elif operation == "Subtraction":
                result_signal = self.subtract_signals()
            elif operation == "Multiplication":
                constant = float(self.const_entry.get())
                result_signal = self.multiply_signal(constant)
            elif operation == "Squaring":
                result_signal = self.square_signal()
            elif operation == "Normalization":
                range_choice = self.norm_var.get()
                result_signal = self.normalize_signal(range_choice)
            elif operation == "Accumulation":
                result_signal = self.accumulate_signal()
            self.plot_signals(result_signal)
        except Exception as e:
            messagebox.showerror("Error", f"Operation failed: {e}")

    def add_signals(self):
        if len(self.signals) < 2:
            raise ValueError("Addition requires at least two signals.")
        result_signal = np.zeros_like(self.signals[0])
        for signal in self.signals:
            result_signal[:, 1] += signal[:, 1]

        indices = self.signals[0][:, 0].astype(int)
        samples = result_signal[:, 1]

        SignalSamplesAreEqual("Signal1+signal2.txt",indices,samples)
        SignalSamplesAreEqual("signal1+signal3.txt",indices,samples)
        return result_signal

    def subtract_signals(self):
        if len(self.signals) < 2:
            raise ValueError("Subtraction requires at least two signals.")
        result_signal = self.signals[0].copy()
        for signal in self.signals[1:]:
            result_signal[:, 1] -= signal[:, 1]
            signal = result_signal[:, 1]
            result_signal[:, 1] = np.abs(signal)

        indices = self.signals[0][:, 0].astype(int)
        samples = result_signal[:, 1]

        SignalSamplesAreEqual("signal1-signal2.txt",indices,samples)
        SignalSamplesAreEqual("signal1-signal3.txt",indices,samples)
        return result_signal

    def multiply_signal(self, constant):
        result_signal = self.signals[0] * constant

        indices = self.signals[0][:, 0].astype(int)
        samples = result_signal[:, 1]

        SignalSamplesAreEqual("MultiplySignalByConstant-Signal1 - by 5.txt",indices,samples)
        SignalSamplesAreEqual("MultiplySignalByConstant-signal2 - by 10.txt",indices,samples)
        return result_signal

    def square_signal(self):
        result_signal = self.signals[0].copy()
        result_signal[:, 1] = np.square(result_signal[:, 1])

        indices = self.signals[0][:, 0].astype(int)
        samples = result_signal[:, 1]

        SignalSamplesAreEqual("Output squaring signal 1.txt",indices,samples)
        return result_signal

    def normalize_signal(self, range_choice):

        signal = self.signals[0].copy()
        y_values = signal[:, 1]

        min_val = np.min(y_values)
        max_val = np.max(y_values)

        if max_val - min_val == 0:
            normalized = np.zeros_like(y_values)
        else:
            if range_choice == "-1 to 1":
                normalized = 2 * (y_values - min_val) / (max_val - min_val) - 1
            elif range_choice == "0 to 1":
                normalized = (y_values - min_val) / (max_val - min_val)

        signal[:, 1] = normalized

        indices = self.signals[0][:, 0].astype(int)
        samples = signal[:, 1]

        SignalSamplesAreEqual("normalize of signal 1 (from -1 to 1)-- output.txt",indices,samples)
        SignalSamplesAreEqual("normlize signal 2 (from 0 to 1 )-- output.txt",indices,samples)

        return signal

    def accumulate_signal(self):
        result_signal = None
        signal = self.signals[0].copy()

        for signal in self.signals:
            y_values = signal[:, 1]
            cumulative_sum = 0
            for i in range(len(y_values)):
                cumulative_sum += y_values[i]
                signal[i, 1] = cumulative_sum

            # Set the result_signal as the current signal
            result_signal = signal

        # Extract indices and samples from result_signal
        indices = result_signal[:, 0].astype(int)
        samples = result_signal[:, 1]

        # Call the test function with result_signal's results
        SignalSamplesAreEqual("output accumulation for signal1.txt", indices, samples)

        return result_signal

    def plot_signals(self, result_signal):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=100)

        # Plot original signals (Continuous and Discrete)
        for signal in self.signals:
            ax1.plot(signal[:, 0], signal[:, 1], label='Original Continuous Signal', linestyle='--', color='gray')
            ax1.stem(signal[:, 0], signal[:, 1], label='Original Discrete Signal', linefmt='C1-', markerfmt='C1o', basefmt='C1-')

        ax1.set_title("Original Signals")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        ax1.legend()

        ax2.plot(result_signal[:, 0], result_signal[:, 1], label='Result Continuous Signal', linestyle='-', color='blue')
        ax2.stem(result_signal[:, 0], result_signal[:, 1], label='Result Discrete Signal', linefmt='C0-', markerfmt='C0o', basefmt='C0-')

        ax2.set_title("Resulting Signal")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Amplitude")
        ax2.legend()

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

def SignalSamplesAreEqual(file_name,indices,samples):
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

    if len(expected_samples)!=len(samples):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    messagebox.showinfo("Success","Test case passed successfully")
