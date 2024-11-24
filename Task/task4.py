import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class TaskPage4(tk.Frame):
    def __init__(self, parent, task_name):
        super().__init__(parent, bg="white")

        self.task_name = task_name
        self.input_file = None
        self.input_samples = None
        self.expected_output_samples = None

        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self, bg="white")
        main_frame.pack(fill="both", expand=True)

        left_frame = tk.Frame(main_frame, bg="white", width=300)
        left_frame.pack(side="left", fill="y", padx=20)

        right_frame = tk.Frame(main_frame, bg="white", width=500)
        right_frame.pack(side="right", fill="both", expand=True)

        title_label = tk.Label(left_frame, text="Frequency Domain", font=("Helvetica", 24, "bold"), bg="white", fg="#2596be")
        title_label.pack(side="top", pady=10)

        self.label = tk.Label(left_frame, text="Step 1: Upload the input signal file", bg="white")
        self.label.pack(pady=10)

        self.upload_button = tk.Button(left_frame, text="Upload Input File", command=self.upload_input_file, width=20, bg="#2596be", fg="white")
        self.upload_button.pack(pady=10)

        self.select_label = tk.Label(left_frame, text="Select DFT, IDFT or DCT", bg="white")
        self.select_label.pack(pady=5)

        self.select_var = tk.StringVar()
        self.select_menu = tk.OptionMenu(left_frame, self.select_var, "", "DFT", "IDFT", "DCT")
        self.select_menu.pack(pady=5)

        self.sampling_freq_label = tk.Label(left_frame, text="Enter the sampling frequency in Hz:", bg="white")
        self.sampling_freq_label.pack(pady=5)

        self.sampling_freq_entry = tk.Entry(left_frame)
        self.sampling_freq_entry.insert(0, "0")
        self.sampling_freq_entry.pack(pady=5)

        self.dct_coefficients_label = tk.Label(left_frame, text="Enter the number of DCT coefficients to save:", bg="white")
        self.dct_coefficients_label.pack(pady=5)

        self.dct_coefficients_entry = tk.Entry(left_frame)
        self.dct_coefficients_entry.insert(0, "0")  # Default to 0
        self.dct_coefficients_entry.pack(pady=5)

        self.process_button = tk.Button(left_frame, text="Process Signal", command=self.process_signal, width=20, bg="#2596be", fg="white")
        self.process_button.pack(pady=10)


        self.time_domain_label = tk.Label(right_frame, text="Time Domain Operations", font=("Helvetica", 20, "bold"), bg="white", fg="#2596be")
        self.time_domain_label.pack(pady=10)

        self.k_label = tk.Label(right_frame, text="Enter k:", bg="white")
        self.k_label.pack(pady=10)
        self.k_entry = tk.Entry(right_frame)
        self.k_entry.pack(pady=5)

        self.operation_var = tk.StringVar()
        self.operation_menu = tk.OptionMenu(right_frame, self.operation_var, "Select Operation", "Sharpen Signal", "Delay Signal","Advance Signal", "Fold Signal","Delay Folded Signal" ,"Advance Folded Signal")
        self.operation_menu.pack(pady=5)

        self.execute_button = tk.Button(right_frame, text="Execute Operation", command=self.execute_operation, width=20, bg="#2596be", fg="white")
        self.execute_button.pack(pady=5)

        self.output_frame = tk.Frame(right_frame, bg="white")
        self.output_frame.pack(pady=10)

    def upload_input_file(self):
        self.input_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if self.input_file:
            self.select_var.set("")
            self.input_samples = None
            messagebox.showinfo("File Uploaded", "Input signal file uploaded successfully!")

    
    def process_signal(self):
        select = self.select_var.get()
        if not select:
            messagebox.showwarning("Error", "Please select either DFT, IDFT, or DCT.")
            return

        if not self.input_file:
            messagebox.showwarning("Error", "Please upload an input signal file.")
            return

        sampling_frequency = float(self.sampling_freq_entry.get())

        if select == "DFT":
            self.input_samples = self.read_file_space2(self.input_file)
        elif select == "IDFT":
            self.input_samples = self.read_file_comma(self.input_file)
        elif select == "DCT":
            self.input_samples = self.read_file_space2(self.input_file)

        if self.input_samples is None:
            messagebox.showerror("Error", "No input samples found.")
            return

        N = len(self.input_samples)
        t = np.linspace(0, 1, N, endpoint=False)


        for widget in self.output_frame.winfo_children():
            widget.destroy()

        if select == "DFT":
            dft = self.compute_fourier(self.input_samples, N, inverse=False)
            amplitude, phase = self.calculate_amplitude_and_phase(dft)
            freqs = [k * sampling_frequency / N for k in range(N)]
            self.plot_frequency_analysis(freqs, amplitude, phase)

        elif select == "IDFT":
            reconstructed_signal = self.compute_fourier(self.input_samples, N, inverse=True)
            self.plot_idft(t, self.input_samples, reconstructed_signal)

        elif select == "DCT":
            dct = self.compute_dct(self.input_samples)
            self.plot_dct(dct)
            m = int(self.dct_coefficients_entry.get())
            self.save_dct_coefficients(dct, m)
          
            

        if self.expected_output_samples:
            comparison_result = self.SignalSamplesAreEqualTk(self.expected_output_samples, self.input_samples)
            messagebox.showinfo("Comparison Result", comparison_result)

    def read_file_space(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        samples = []
        index =[]
        for line in lines[3:]:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    x = float(parts[0])
                    value = float(parts[1])
                    samples.append(value)
                    index.append(x)
                except ValueError:
                    continue
        return samples ,index
    
    def read_file_space2(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        samples = []
        index =[]
        for line in lines[3:]:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    x = float(parts[0])
                    value = float(parts[1])
                    samples.append(value)
                    index.append(x)
                except ValueError:
                    continue
        return samples 

    def read_file_comma(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        samples = []
        for line in lines[4:]:
            if line.strip():
                parts = line.split(',')
                if len(parts) == 2:
                    try:
                        amplitude = float(parts[0].replace('f', ''))
                        phase = float(parts[1].replace('f', ''))
                        real = amplitude * np.cos(phase)
                        imag = amplitude * np.sin(phase)
                        complex_value = real + 1j * imag
                        samples.append(complex_value)
                    except ValueError:
                        continue
        return samples 

    def compute_fourier(self, input_samples, N, inverse=False):
        fourier_result = []
        for k in range(N):
            real_sum = 0
            imag_sum = 0
            for n in range(N):
                angle = 2 * np.pi * k * n / N
                real_sum += input_samples[n] * np.cos(angle)
                imag_sum += (input_samples[n] * np.sin(angle) if inverse else -input_samples[n] * np.sin(angle))

            if inverse:
                real_sum /= N
                imag_sum /= N

            X_k = real_sum + 1j * imag_sum
            fourier_result.append(X_k)

        return fourier_result

    def compute_dct(self, input_samples):
        N = len(input_samples)
        dct_result = np.zeros(N)
        factor = np.sqrt(2 / N)

        for k in range(N):
            sum_value = 0
            for n in range(0, N ):
                sum_value += input_samples[n] * np.cos(np.pi / (4 * N) * (2 * n - 1) * (2 * k - 1))
            dct_result[k] = factor * sum_value
        t = np.linspace(0, 1, len(input_samples), endpoint=False)
        self.SignalSamplesAreEqual("DCT_output.txt",0,dct_result)
        
       

        return dct_result

    def calculate_amplitude_and_phase(self, fourier_result):
        amplitude = [np.abs(X_k) for X_k in fourier_result]
        phase = [np.arctan2(X_k.imag, X_k.real) for X_k in fourier_result]
        return amplitude, phase

    def plot_frequency_analysis(self, freqs, amplitude, phase):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].stem(freqs, amplitude, basefmt=" ", linefmt='blue', markerfmt='bo', label="Discrete")
        axs[0].set_title('Frequency vs Amplitude')
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_xlim(0, np.max(freqs))

        axs[1].stem(freqs, phase, basefmt=" ", linefmt='orange', markerfmt='ro', label="Discrete")
        axs[1].set_title('Frequency vs Phase')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Phase (radians)')
        axs[1].set_xlim(0, np.max(freqs))

        canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()


    def execute_operation(self):
        operation = self.operation_var.get()
        if operation == "Sharpen Signal":
            self.SharpeningFunction()
        elif operation == "Delay Signal":
            delayed_x,delayed_signal=self.delay_signal() 
            self.Shift_Fold_Signal("output shifting by minus 500.txt",delayed_x,delayed_signal) 
            
        elif operation == "Fold Signal":
            self.fold_signal()
        elif operation == "Advance Signal":
            advanced_x,advanced_signal=self.advance_signal()
            self.Shift_Fold_Signal("output shifting by add 500.txt",advanced_x,advanced_signal)
        elif operation == "Advance Folded Signal":
            advanced_x,advanced_signal=self.advance_signal()
            self.Shift_Fold_Signal("Output_ShiftFoldedby-500.txt",advanced_x,advanced_signal)
        elif operation == "Delay Folded Signal":
             delayed_x,delayed_signal=self.delay_signal() 
             self.Shift_Fold_Signal("Output_ShifFoldedby500.txt",delayed_x,delayed_signal)      
        else:
            messagebox.showwarning("Error", "Please select a valid operation.")

    def SharpeningFunction(self):
        for widget in self.output_frame.winfo_children():
            widget.destroy()
        InputSignal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

        def DerivativeSignal(InputSignal):
            expectedOutput_first = [1] * 99
            expectedOutput_second = [0] * 98

            FirstDrev = []
            SecondDrev = []

            for n in range(1, len(InputSignal)):
                 FirstDrev.append(InputSignal[n] - InputSignal[n - 1])
                 
            for n in range(1, len(InputSignal) - 1):
                SecondDrev.append(InputSignal[n + 1] - 2 * InputSignal[n] + InputSignal[n - 1])
            print(FirstDrev)
            print(SecondDrev)
            # Testing your code
            if (len(FirstDrev) != len(expectedOutput_first)) or (len(SecondDrev) != len(expectedOutput_second)):
                print("Mismatch in length")
                return None, None
            first = second = True
            for i in range(len(expectedOutput_first)):
                if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
                    continue
                else:
                    first = False
                    print("1st derivative wrong")
                    return None, None
            for i in range(len(expectedOutput_second)):
                if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
                    continue
                else:
                    second = False
                    print("2nd derivative wrong")
                    return None, None
            if first and second:
                messagebox.showinfo("Sucess","Test case passed successfully")
            else:
                messagebox.showerror("Failed","Derivative Test case failed")
            return FirstDrev, SecondDrev

        def Sharpen(InputSignal, first_derivative, second_derivative):
            sharpened_signal = []
            for i in range(1, len(InputSignal) - 1):
                sharpened_value = InputSignal[i] + 0.5 * first_derivative[i - 1] + 0.25 * second_derivative[i - 1]
                sharpened_signal.append(sharpened_value)
               
            return sharpened_signal

        FirstDrev, SecondDrev = DerivativeSignal(InputSignal)

        if FirstDrev is not None and SecondDrev is not None:
            sharpened_signal = Sharpen(InputSignal, FirstDrev, SecondDrev)
            t = np.linspace(0, 1, len(sharpened_signal), endpoint=False)
            self.plot_signal(t, sharpened_signal, "Sharpened Signal")
            return sharpened_signal
        else:
            return None

    def delay_signal(self):
            for widget in self.output_frame.winfo_children():
                widget.destroy()
            if not self.input_file:
                messagebox.showerror("Error", "Please upload an input signal file.")
                return
            self.input_samples, x = self.read_file_space(self.input_file)

           
            if self.input_samples is None:
                messagebox.showerror("Error", "No input samples found.")
                return
            try:
              k = int(self.k_entry.get())
              k *= 1
            except ValueError as e:
               messagebox.showerror("Error", str(e))
               return
           

            delayed_x = [value + k for value in x]
            delayed_signal = self.input_samples[:]
            self.plot_signal(delayed_x,  delayed_signal , "delay Signal")
            
            return delayed_x, delayed_signal

    def fold_signal(self):
        for widget in self.output_frame.winfo_children():
            widget.destroy()
        if not self.input_file:
            messagebox.showerror("Error", "Please upload an input signal file.")
            return
        self.input_samples ,x = self.read_file_space(self.input_file)
        if self.input_samples is None:
            messagebox.showerror("Error", "No input samples found.")
            return
        # folded = self.input_samples[::-1]
        folded_x = [-value for value in reversed(x)]
        folded_y = list(reversed(self.input_samples ))

        # t = np.linspace(0, 1000, len(folded), endpoint=False)
        self.plot_signal(folded_x, folded_y, "Folded Signal")
        self.Shift_Fold_Signal("Output_fold.txt",folded_x,folded_y)

    def advance_signal(self):
        for widget in self.output_frame.winfo_children():
            widget.destroy()
        if not self.input_file:
            messagebox.showerror("Error", "Please upload an input signal file.")
            return
        self.input_samples , x = self.read_file_space(self.input_file)
        if self.input_samples is None:
            messagebox.showerror("Error", "No input samples found.")
            return
        try:
            k = int(self.k_entry.get())
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        if k >= len(self.input_samples):
            messagebox.showerror("Error", "k must be less than the number of input samples.")
            return

        advanced_x = [value - k for value in x]
        advanced_signal = self.input_samples[:]
        self.plot_signal(advanced_x,  advanced_signal , "advanced Signal")
        
        return advanced_x, advanced_signal
       

    def plot_signal(self, t, signal, title):
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(t, signal)
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
       
        canvas = FigureCanvasTkAgg(fig, self.output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    def plot_idft(self, t, input_samples, reconstructed_signal):
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(t, np.real(reconstructed_signal), label='Reconstructed Signal', color='blue')
        ax.plot(t, input_samples, label='Original Signal', color='orange', linestyle='dashed')
        ax.set_title('IDFT Signal Reconstruction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def plot_dct(self, dct):
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(dct, label='DCT Coefficients', color='green')
        ax.set_title('Discrete Cosine Transform (DCT)')
        ax.set_xlabel('Index')
        ax.set_ylabel('Coefficient Value')
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def save_dct_coefficients(self, dct_result, m):
        if m > 0:
            with open("dct_coefficients.txt", "w") as f:
                for i in range(min(m, len(dct_result))):
                    f.write(f"0 {dct_result[i]}\n")
            messagebox.showinfo("Success", f"Saved the first {m} DCT coefficients to 'dct_coefficients.txt'.")
        else:
            messagebox.showwarning("Invalid Input", "Please enter a valid number of coefficients.")

    # def SignalSamplesAreEqualTk(self, expected_samples, input_samples):
    #     equal = sc.SignalSamplesAreEqual(expected_samples, input_samples)
    #     return "The samples are equal." if equal else "The samples are not equal."

    def SignalSamplesAreEqual(self,file_name,indices,samples):
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
            messagebox.showerror("Failed","Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(expected_samples)):
            if abs(samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                messagebox.showerror("Failed","Test case failed, your signal have different length from the expected one")
                return
        messagebox.showinfo("Sucess","Test case passed successfully")

    def Shift_Fold_Signal(self,file_name ,Your_indices,Your_samples):
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
            messagebox.showerror("Failed","Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(Your_indices)):
            if(Your_indices[i]!=expected_indices[i]):
                messagebox.showerror("Failed","Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
                return
        for i in range(len(expected_samples)):
            if abs(Your_samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                messagebox.showerror("Failed","Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
                return
        messagebox.showinfo("Sucess","Shift_Fold_Signal Test case passed successfully")