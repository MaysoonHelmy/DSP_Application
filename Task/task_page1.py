import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.fft import fft

class TaskPage1(tk.Frame):
    def __init__(self, parent, task_name):
        super().__init__(parent, bg="#FFFFFF")
        self.task_name = task_name

        # Page title
        title_label = tk.Label(self, text=task_name, font=("Helvetica", 24, "bold"), bg="#FFFFFF")
        title_label.pack(pady=20)

        # Button to load the signal from a file
        load_button = ttk.Button(self, text="Read Signal", command=self.load_signal, width=20)
        load_button.pack(pady=10)

        # Button to generate a signal
        generate_button = ttk.Button(self, text="Generate Signal", command=self.show_input_window, width=20)
        generate_button.pack(pady=10)

        # Frame for the plots (two columns)
        self.plot_frame = tk.Frame(self, bg="#FFFFFF")
        self.plot_frame.pack(fill="both", expand=True)

        # Frames for time and frequency domain plots
        self.time_frame = tk.Frame(self.plot_frame, bg="#FFFFFF")
        self.time_frame.pack(side="top", fill="both", expand=True, padx=5)

        self.freq_frame = tk.Frame(self.plot_frame, bg="#FFFFFF")
        self.freq_frame.pack(side="bottom", fill="both", expand=True, padx=5)

    def load_signal(self):
        file_path = filedialog.askopenfilename(title="Open Signal File", filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    signal_lines = file.readlines()

                if len(signal_lines) < 3:
                    raise ValueError("The file must contain at least three lines for signal processing.")

                # Read signal type
                signal_type = int(signal_lines[0].strip())
                if signal_type not in [0, 1]:
                    raise ValueError("Signal type must be either 0 (Time Domain) or 1 (Frequency Domain).")

                # Read periodicity (not used later in your method, consider removing if not needed)
                is_periodic = int(signal_lines[1].strip())

                # Read number of samples
                N1 = int(signal_lines[2].strip())
                if N1 <= 0:
                    raise ValueError("Number of samples or frequencies must be a positive integer.")

                # Read data
                data_lines = signal_lines[3:3 + N1]
                data = np.array([list(map(float, line.split())) for line in data_lines if line.strip()])

                # Plot based on signal type
                if signal_type == 0:
                    self.plot_time_domain(data)
                elif signal_type == 1:
                    self.plot_frequency_domain(data)

            except ValueError as ve:
                messagebox.showerror("Error", f"Error loading signal: {ve}")
            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error: {e}")

    def plot_time_domain(self, data):
        # Clear previous plot if any
        for widget in self.time_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)

        ax.plot(data[:, 0], data[:, 1], label='Continuous Signal', color='blue', linewidth=1)
        ax.scatter(data[:, 0], data[:, 1], label='Discrete Samples', color='red', marker='o')

        ax.set_title('Time Domain Representation', fontsize=14)
        ax.set_xlabel('Sample Index', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
        ax.set_xlim(0, data[:, 0].max())
        ax.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)
        ax.legend(loc='upper right', fontsize=12)

        canvas = FigureCanvasTkAgg(fig, master=self.time_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=150, pady=10)

    def plot_frequency_domain(self, data):
        # Clear previous plot if any
        for widget in self.freq_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)

        if data.shape[1] == 1:  # If data has only one column (time domain)
            fft_data = fft(data[:, 0])
            magnitude = np.abs(fft_data)
            frequencies = np.fft.fftfreq(len(fft_data)) * len(fft_data)
            ax.stem(frequencies, magnitude, basefmt=" ", linefmt='b-', markerfmt='bo', label='Discrete Samples')
            ax.plot(frequencies, magnitude, color='r', label='Continuous Signal')
        else:
            ax.stem(data[:, 0], data[:, 1], basefmt=" ", linefmt='b-', markerfmt='bo', label='Discrete Samples')
            ax.plot(data[:, 0], data[:, 1], color='r', label='Continuous Signal')

        ax.set_title('Frequency Domain Representation', fontsize=14)
        ax.set_xlabel('Frequency (Hz)', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
        ax.set_xlim(min(data[:, 0]), max(data[:, 0]))  # Set x-axis limits to the minimum and maximum frequencies
        ax.set_ylim(min(data[:, 1]) - 1, max(data[:, 1]) + 1)  # Set y-axis limits to the minimum and maximum amplitudes
        ax.legend(loc='upper right', fontsize=12)

        canvas = FigureCanvasTkAgg(fig, master=self.freq_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=100, pady=10)

    def show_input_window(self):
        """Show input window for signal generation."""
        input_window = tk.Toplevel(self)
        input_window.title("Generate Signal")
        input_window.geometry("600x500")

        # Create a frame for input fields (left side)
        input_frame = tk.Frame(input_window)
        input_frame.pack(side="left", padx=20, pady=20)

        # Create a frame for the plot (right side)
        self.plot_area = tk.Frame(input_window, width=300, height=400)
        self.plot_area.pack(side="right", padx=20, pady=20)

        # Wave Type Selection
        tk.Label(input_frame, text="Select Wave Type:", font=("Helvetica", 15)).pack(pady=12)
        self.wave_type_var = tk.StringVar(value="Sine")
        sine_radio = tk.Radiobutton(input_frame, text="Sine Wave", variable=self.wave_type_var, value="Sine", font=("Helvetica", 12))
        sine_radio.pack()
        cosine_radio = tk.Radiobutton(input_frame, text="Cosine Wave", variable=self.wave_type_var, value="Cosine", font=("Helvetica", 12))
        cosine_radio.pack()

        # Amplitude Input
        tk.Label(input_frame, text="Amplitude (A):", font=("Helvetica", 15)).pack(pady=5)
        self.amplitude_entry = tk.Entry(input_frame, font=("Helvetica", 12))
        self.amplitude_entry.pack(pady=5)

        # Phase Shift Input
        tk.Label(input_frame, text="Phase Shift (theta in radians):", font=("Helvetica", 15)).pack(pady=5)
        self.phase_entry = tk.Entry(input_frame, font=("Helvetica", 12))
        self.phase_entry.pack(pady=5)

        # Frequency Input
        tk.Label(input_frame, text="Frequency (Hz):", font=("Helvetica", 15)).pack(pady=5)
        self.frequency_entry = tk.Entry(input_frame, font=("Helvetica", 12))
        self.frequency_entry.pack(pady=5)

        # Sampling Frequency Input
        tk.Label(input_frame, text="Sampling Frequency (Hz):", font=("Helvetica", 15)).pack(pady=5)
        self.sampling_entry = tk.Entry(input_frame, font=("Helvetica", 12))
        self.sampling_entry.pack(pady=5)

        # End Time Input
        # tk.Label(input_frame, text="End Time (s):", font=("Helvetica", 15)).pack(pady=5)
        #self.end_time_entry = tk.Entry(input_frame, font=("Helvetica", 12))
        # self.end_time_entry.insert(0, "1")
        # self.end_time_entry.pack(pady=5)

        # Show Both Signals Button
        show_both_button = tk.Button(input_frame, text="Show Both Signals", command=lambda: self.show_both_signals(), font=("Helvetica", 12), bg="#3498DB", fg="white")
        show_both_button.pack(pady=12)

        # Generate Signal Button
        generate_button = tk.Button(input_frame, text="Generate Signal", command=self.generate_signal_with_wave_type, width=20, font=("Helvetica", 12), bg="#3498DB", fg="white")
        generate_button.pack(pady=12)

    def generate_signal_with_wave_type(self):
        wave_type = self.wave_type_var.get()
        self.generate_signal(wave_type)

    def generate_signal(self, wave_type):
        """Generate the selected signal and plot it."""
        try:
            A = float(self.amplitude_entry.get())
            theta = float(self.phase_entry.get())
            f = float(self.frequency_entry.get())
            fs = float(self.sampling_entry.get())
            # end_time=float(self.end_time_entry.get())
            end_time=1

            # Check sampling theorem
            if fs < 2 * f:
                messagebox.showerror("Error", "Sampling frequency must be at least twice the frequency (Nyquist criterion).")
                return

            # Time array
            t = np.arange(0,end_time, 1/fs)  # Small time window for better visualization

            # Generate the selected signal
            if wave_type == "Sine":
                signal = A * np.sin(2 * np.pi * f * t + theta)
                title = "Sine Wave Signal"
                color = 'b'  # blue
                SignalSamplesAreEqual("SinOutput.txt",t, signal)
            else:
                signal = A * np.cos(2 * np.pi * f * t + theta)
                title = "Cosine Wave Signal"
                color = 'r'  # red
                SignalSamplesAreEqual("CosOutput.txt",t, signal)

            # Clear previous plot
            for widget in self.plot_area.winfo_children():
                widget.destroy()

            # Create a new figure and plot the selected signal
            fig = Figure(figsize=(6, 4), dpi=120)  # Increased figure size
            ax = fig.add_subplot(111)
            ax.plot(t, signal, color=color, linewidth=2)  # Increased line width
            ax.set_title(title, fontsize=16)  # Increased title font size
            ax.set_xlabel("Time (s)", fontsize=15)  # Increased x-axis label font size
            ax.set_xlim(0, 0.01)
            ax.set_ylabel("Amplitude", fontsize=15)  # Increased y-axis label font size
            ax.grid()

            # Create a canvas to display the plot
            canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
            canvas.draw()
            canvas.get_tk_widget().pack()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values .")

    def show_both_signals(self):
        """Generate both sine and cosine signals and plot them together."""
        try:
            A = float(self.amplitude_entry.get())
            theta = float(self.phase_entry.get())
            f = float(self.frequency_entry.get())
            fs = float(self.sampling_entry.get())
            #end_time=float(self.end_time_entry.get())
            end_time=1
            num_samples=int(fs*1)
            samples=np.arange(num_samples)

            # Check sampling theorem
            if fs < 2 * f:
                messagebox.showerror("Error", "Sampling frequency must be at least twice the frequency (Nyquist criterion).")
                return

            # Time array
            t = np.arange(0,end_time, 1/fs)  # Small time window for better visualization
            #t=samples/fs

            # Generate both sine and cosine signals
            sine_signal = A * np.sin(2 * np.pi * f * t + theta)


            cosine_signal = A * np.cos(2 * np.pi * f * t + theta)

            # Clear previous plot
            for widget in self.plot_area.winfo_children():
                widget.destroy()

            # Create a new figure and plot both signals
            fig = Figure(figsize=(6, 4), dpi=120)  # Increased figure size
            ax = fig.add_subplot(111)
            ax.plot(t, sine_signal, label="Sine Wave", color='b', linewidth=2)  # Increased line width
            ax.plot(t, cosine_signal, label="Cosine Wave", color='r', linewidth=2)  # Increased line width
            ax.set_title("Sine and Cosine Waves", fontsize=16)  # Increased title font size
            ax.set_xlabel("Time (s)", fontsize=15)  # Increased x-axis label font size
            ax.set_xlim(0, 0.01)
            ax.set_ylabel("Amplitude", fontsize=15)  # Increased y-axis label font size
            ax.grid()
            ax.legend(fontsize=12)

            # Create a canvas to display the plot
            canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
            canvas.draw()
            canvas.get_tk_widget().pack()
        # SignalSamplesAreEqual("SinOutput.txt",t, sine_signal)
        # SignalSamplesAreEqual("CosOutput.txt",t, cosine_signal)

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values.")

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
    print("Test case passed successfully")