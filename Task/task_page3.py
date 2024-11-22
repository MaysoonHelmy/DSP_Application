import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TaskPage3(tk.Frame):
    def __init__(self, parent, task_name):
        super().__init__(parent, bg="#FFFFFF")
        self.task_name = task_name
        self.signal = None  # Placeholder for the signal data

        # Page Title
        title_label = tk.Label(self, text=task_name, font=("Helvetica", 24, "bold"), bg="#FFFFFF")
        title_label.pack(pady=20)

        # Option to select between "Number of Bits" and "Number of Levels"
        option_label = tk.Label(self, text="Select Quantization Method:", font=("Helvetica", 14), bg="#FFFFFF")
        option_label.pack(pady=5)

        self.quantization_method = tk.StringVar(value="Bits")
        bits_radio = tk.Radiobutton(self, text="Number of Bits", variable=self.quantization_method, value="Bits", font=("Helvetica", 12), bg="#FFFFFF", command=self.show_input_box)
        bits_radio.pack()
        levels_radio = tk.Radiobutton(self, text="Number of Levels", variable=self.quantization_method, value="Levels", font=("Helvetica", 12), bg="#FFFFFF", command=self.show_input_box)
        levels_radio.pack()

        # Input Box for Bits/Levels
        self.input_frame = tk.Frame(self, bg="#FFFFFF")
        self.input_frame.pack(pady=10)

        # Input field for bits/levels
        self.param_label = tk.Label(self.input_frame, text="Enter Number of Bits:", font=("Helvetica", 12), bg="#FFFFFF")
        self.param_label.grid(row=0, column=0, padx=5, pady=5)
        self.param_entry = tk.Entry(self.input_frame, font=("Helvetica", 12))
        self.param_entry.grid(row=0, column=1, padx=5, pady=5)

        # Load Signal Button
        load_button = tk.Button(self, text="Load Signal", command=self.load_signal, font=("Helvetica", 14), bg="#3498DB", fg="#FFFFFF")
        load_button.pack(pady=20)

        # Quantize Button
        quantize_button = tk.Button(self, text="Quantize Signal", command=self.quantize_signal, font=("Helvetica", 14), bg="#3498DB", fg="#FFFFFF")
        quantize_button.pack(pady=20)

        # Output Text Box for displaying results in table format
        self.output_text = tk.Text(self, wrap='none', height=10, width=80, font=("Courier", 12))  # Set width for table-like formatting
        self.output_text.pack(pady=20)

        # Add horizontal and vertical scrollbars for better viewing
        self.x_scrollbar = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.output_text.xview)
        self.x_scrollbar.pack(fill=tk.X, side=tk.BOTTOM)
        self.output_text.config(xscrollcommand=self.x_scrollbar.set)

        # Frame for Matplotlib plot
        self.plot_frame = tk.Frame(self, bg="#FFFFFF")
        self.plot_frame.pack(pady=10)

    def show_input_box(self):
        if self.quantization_method.get() == "Bits":
            self.param_label.config(text="Enter Number of Bits:")
        else:
            self.param_label.config(text="Enter Number of Levels:")

    def load_signal(self):
        """Load signal data from a selected file."""
        file_path = filedialog.askopenfilename(title="Open Signal File", filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    signal_lines = file.readlines()

                # Reading the number of samples and signal values from the file
                N1 = int(signal_lines[2].strip())  # Number of samples
                data_lines = signal_lines[3:3 + N1]
                self.signal = np.array([float(line.split()[1]) for line in data_lines])

                messagebox.showinfo("Success", "Signal loaded successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load signal: {e}")

    def quantize_signal(self):
        if self.signal is None:
            messagebox.showerror("Error", "Please load a signal file first.")
            return

        try:
            # Read number of bits or levels
            if self.quantization_method.get() == "Bits":
                num_bits = int(self.param_entry.get())
                num_levels = 2 ** num_bits
            else:
                num_levels = int(self.param_entry.get())
                num_bits = int(np.ceil(np.log2(num_levels)))

            # Perform quantization using midpoint, but take lower interval on boundary
            min_val, max_val = min(self.signal), max(self.signal)
            interval_size = (max_val - min_val) / num_levels
            quantized_signal = []
            encoded = []
            errors = []
            interval_indices = []

            for i, value in enumerate(self.signal):
                # Find the quantization interval index
                normalized_value = (value - min_val) / interval_size
                interval_index = int(normalized_value)  # Get the lower interval

                if i == len(self.signal) - 1:
                    quantized_value = min_val + (interval_index + 1) * interval_size  # Upper midpoint
                else:
                  # If value is exactly on a boundary, ensure we take the lower interval
                      if normalized_value == interval_index:
                             interval_index -= 1  # Move to the lower interval if exactly on boundary
                # Compute the midpoint of the quantization interval
                quantized_value = min_val + (interval_index + 0.5) * interval_size  # Midpoint
                quantized_value = max(quantized_value, 0)
                binary_code = format(interval_index, f'0{num_bits}b')
                error = value - quantized_value

                # Format with three significant figures and append to lists
                quantized_signal.append(round(quantized_value, 2))  # Store with 2 significant figures
                encoded.append(binary_code)
                errors.append(round(error, 2))  # Store with 2 significant figures
                interval_indices.append(interval_index)

            # Call test functions with the original signal and quantized signal
            test_results_1 = self.QuantizationTest1("Quan1_Out.txt", encoded, quantized_signal)
            test_results_2 = self.QuantizationTest2("Quan2_Out.txt", interval_indices, encoded, quantized_signal, errors)

            # Display results in table format
            self.display_results(self.signal, quantized_signal, encoded, errors, interval_indices)  # Original signal first

            # Plot original and quantized signals
            self.plot_signals(self.signal, quantized_signal)  # Plot original first, quantized second

            # Show test results in a message box
            test_results_message = f"{test_results_1}\n{test_results_2}"
            messagebox.showinfo("Test Results", test_results_message)

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values.")

    def display_results(self, original_signal, quantized_signal, encoded, errors, interval_indices):
        # Clear the text widget
        self.output_text.delete(1.0, tk.END)

        # Table header
        header = f"{'Original Signal':^20}{'Interval Index':^20}{'Quantized Signal':^20}{'Encoded Signal':^20}{'Quantization Error':^20}\n"
        self.output_text.insert(tk.END, header)
        self.output_text.insert(tk.END, '-' * 80 + '\n')

        # Table content
        for i in range(len(original_signal)):
            line = f"{original_signal[i]:^20.3f}{interval_indices[i]:^20}{quantized_signal[i]:^20.3f}{encoded[i]:^20}{errors[i]:^20.3f}\n"
            self.output_text.insert(tk.END, line)

        # Scroll to the top to view results
        self.output_text.yview_moveto(0)

    def plot_signals(self, original_signal, quantized_signal):
        # Clear the previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Create a new figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(original_signal, label="Original Signal", marker='o')
        ax.plot(quantized_signal, label="Quantized Signal", marker='x', linestyle='--')

        ax.set_title("Original and Quantized Signal")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.legend()

        # Embed the plot into the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def QuantizationTest1(self, file_name, Your_EncodedValues, Your_QuantizedValues):
        expectedEncodedValues = []
        expectedQuantizedValues = []
        with open(file_name, 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            while True:
                line = f.readline().strip()
                if len(line.split(' ')) == 2:
                    L = line.split(' ')
                    V2 = str(L[0])
                    V3 = float(L[1])
                    expectedEncodedValues.append(V2)
                    expectedQuantizedValues.append(V3)
                else:
                    break

        if len(Your_EncodedValues) != len(expectedEncodedValues) or len(Your_QuantizedValues) != len(expectedQuantizedValues):
            return "Test Failed: Number of Encoded values or Quantized values does not match the expected output."

        for i in range(len(Your_EncodedValues)):
            if Your_EncodedValues[i] != expectedEncodedValues[i]:
                return f"Test Failed: Encoded value at index {i} does not match the expected output."

        for i in range(len(Your_QuantizedValues)):
            if Your_QuantizedValues[i] != expectedQuantizedValues[i]:
                return f"Test Failed: Quantized value at index {i} does not match the expected output."

        return "Test Passed: Quantization Test 1 passed!"

    def QuantizationTest2(self, file_name, interval_indices, encoded_values, quantized_values, errors):
        expected_indices = []
        expected_encoded = []
        expected_quantized = []
        expected_errors = []
        with open(file_name, 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            while True:
                line = f.readline().strip()
                if len(line.split(' ')) == 4:
                    L = line.split(' ')
                    V1 = int(L[0])
                    V2 = str(L[1])
                    V3 = float(L[2])
                    V4 = float(L[3])
                    expected_indices.append(V1)
                    expected_encoded.append(V2)
                    expected_quantized.append(V3)
                    expected_errors.append(V4)
                else:
                    break

        if (len(interval_indices) != len(expected_indices)) or (len(encoded_values) != len(expected_encoded)) or (len(quantized_values) != len(expected_quantized)) or (len(errors) != len(expected_errors)):
            return "Test Failed: One of the lists has a different size than expected."

        for i in range(len(interval_indices)):
            if interval_indices[i] != expected_indices[i]:
                return f"Test Failed: Interval index at position {i} does not match the expected output."

        for i in range(len(encoded_values)):
            if encoded_values[i] != expected_encoded[i]:
                return f"Test Failed: Encoded value at position {i} does not match the expected output."

        for i in range(len(quantized_values)):
            if quantized_values[i] != expected_quantized[i]:
                return f"Test Failed: Quantized value at position {i} does not match the expected output."

        for i in range(len(errors)):
            if errors[i] != expected_errors[i]:
                return f"Test Failed: Error value at position {i} does not match the expected output."

        return "Test Passed: Quantization Test 2 passed!"
