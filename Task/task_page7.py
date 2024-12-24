import tkinter as tk
from tkinter import ttk ,filedialog, messagebox
import numpy as np
import math
from CompareSignal import Compare_Signals


class Task7(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="#FFFFFF")
        self.create_widgets()

    def create_widgets(self):
        self.title_label = tk.Label(self, text="FIR and Resampling", bg="#FFFFFF", font=("Arial", 20, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="nsew")

        self.fir_frame = tk.Frame(self, bg="#FFFFFF", relief=tk.RIDGE, borderwidth=2)
        self.fir_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.filter_type_label = tk.Label(self.fir_frame, text="Filter Type", bg="#FFFFFF")
        self.filter_type_label.grid(row=0, column=0, padx=5, pady=5)
        self.filter_type = ttk.Combobox(self.fir_frame, values=['Low pass', 'High pass', 'Band pass', 'Band stop'])
        self.filter_type.grid(row=0, column=1, padx=5, pady=5)

        self.fs_label = tk.Label(self.fir_frame, text="Sampling Frequency (Fs)", bg="#FFFFFF")
        self.fs_label.grid(row=1, column=0, padx=5, pady=5)
        self.fs_entry = tk.Entry(self.fir_frame)
        self.fs_entry.grid(row=1, column=1, padx=5, pady=5)

        self.fc_label = tk.Label(self.fir_frame, text="Cutoff Frequency 1 (Fc)", bg="#FFFFFF")
        self.fc_label.grid(row=2, column=0, padx=5, pady=5)
        self.fc_entry = tk.Entry(self.fir_frame)
        self.fc_entry.grid(row=2, column=1, padx=5, pady=5)

        self.fc2_label = tk.Label(self.fir_frame, text="Cutoff Frequency 2 (Fc2)", bg="#FFFFFF")
        self.fc2_label.grid(row=3, column=0, padx=5, pady=5)
        self.fc2_entry = tk.Entry(self.fir_frame)
        self.fc2_entry.grid(row=3, column=1, padx=5, pady=5)

        self.stop_band_label = tk.Label(self.fir_frame, text="Stop Band Attenuation", bg="#FFFFFF")
        self.stop_band_label.grid(row=4, column=0, padx=5, pady=5)
        self.stop_band_entry = tk.Entry(self.fir_frame)
        self.stop_band_entry.grid(row=4, column=1, padx=5, pady=5)

        self.transition_band_label = tk.Label(self.fir_frame, text="Transition Band", bg="#FFFFFF")
        self.transition_band_label.grid(row=5, column=0, padx=5, pady=5)
        self.transition_band_entry = tk.Entry(self.fir_frame)
        self.transition_band_entry.grid(row=5, column=1, padx=5, pady=5)

        self.calculate_button = tk.Button(self.fir_frame, text="Calculate Filter", command=self.calculate_filter)
        self.calculate_button.grid(row=6, column=0, columnspan=2, padx=5, pady=10)

        self.load_button = tk.Button(self.fir_frame, text="Load Signal", command=self.load_files)
        self.load_button.grid(row=7, column=0, columnspan=2, padx=5, pady=10)

        self.convolve_button = tk.Button(self.fir_frame, text="Convolve Signal", command=self.convolution)
        self.convolve_button.grid(row=8, column=0, columnspan=2, padx=5, pady=10)

        self.result_label = tk.Label(self, text="Filter Coefficients:", bg="#FFFFFF")
        self.result_label.grid(row=2, column=0, padx=5, pady=5)

        self.result_text = tk.Text(self, height=10, width=40)
        self.result_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        self.resampling_frame = tk.Frame(self, bg="#FFFFFF", relief=tk.RIDGE, borderwidth=2)
        self.resampling_frame.grid(row=1, column=10, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.resample_label = tk.Label(self.resampling_frame, text="Resampling", bg="#FFFFFF")
        self.resample_label.grid(row=0, column=0, columnspan=2, pady=5)

        self.resample_l_factor_label = tk.Label(self.resampling_frame, text="Resampling Factor L", bg="#FFFFFF")
        self.resample_l_factor_label.grid(row=1, column=0, padx=5, pady=5)
        self.resample_l_factor_entry = tk.Entry(self.resampling_frame)
        self.resample_l_factor_entry.grid(row=1, column=1, padx=5, pady=5)

        self.resample_factor_label = tk.Label(self.resampling_frame, text="Resampling Factor M", bg="#FFFFFF")
        self.resample_factor_label.grid(row=2, column=0, padx=5, pady=5)
        self.resample_factor_entry = tk.Entry(self.resampling_frame)
        self.resample_factor_entry.grid(row=2, column=1, padx=5, pady=5)

        self.resample_button = tk.Button(self.resampling_frame, text="Resample Signal", command=self.resample_signal)
        self.resample_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

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

    def convolution(self):
        try:
            if not hasattr(self, 'signals') or not self.signals:
                messagebox.showerror("Error", "No signals loaded.")
                return

            filter_type = self.filter_type.get()
            input_signal = self.signals[0]
            input_indices = input_signal['indices']
            input_samples = input_signal['samples']

            filtered_signal = self.filtered_signal
            filtered_indices = filtered_signal['indices']
            filtered_samples = filtered_signal['samples']

            min_index = input_indices[0] + filtered_indices[0]
            max_index = input_indices[-1] + filtered_indices[-1]
            output_indices = list(range(min_index, max_index + 1))

            output_samples = []

            for n in output_indices:
                sum_value = 0
                for k in range(len(input_samples)):
                    x_index = input_indices[k]
                    h_index = n - x_index
                    h_indices = np.where(filtered_indices == h_index)[0]

                    if len(h_indices) > 0:
                        h_value = filtered_samples[h_indices[0]]
                    else:
                        h_value = 0

                    sum_value += input_samples[k] * h_value
                output_samples.append(sum_value)

            self.convolution_result = {'indices': output_indices, 'samples': output_samples}

            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "Convolution Result:\n")
            for idx, sample in zip(output_indices, output_samples):
                self.result_text.insert(tk.END, f"Index: {idx}, Sample: {sample:.6f}\n")

            if filter_type == 'Low pass':
                Compare_Signals("FIR Test cases/ecg_low_pass_filtered.txt", self.convolution_result['indices'], self.convolution_result['samples'])
            elif filter_type == 'High pass':
                Compare_Signals("FIR Test cases/ecg_high_pass_filtered.txt", self.convolution_result['indices'], self.convolution_result['samples'])
            elif filter_type == 'Band pass':
                Compare_Signals("FIR Test cases/ecg_band_pass_filtered.txt", self.convolution_result['indices'], self.convolution_result['samples'])
            elif filter_type == 'Band stop':
                Compare_Signals("FIR Test cases/ecg_band_stop_filtered.txt", self.convolution_result['indices'], self.convolution_result['samples'])

            return self.convolution_result

        except Exception as e:
            messagebox.showerror("Error", f"Error during convolution: {e}")

    def calculate_filter(self):
        try:
            filter_type = self.filter_type.get()
            fs = float(self.fs_entry.get())
            fc = float(self.fc_entry.get())
            fc2 = float(self.fc2_entry.get()) if self.fc2_entry.get() else None
            stop_band_attenuation = float(self.stop_band_entry.get())
            transition_band = float(self.transition_band_entry.get())

            # Design the FIR filter and get indices and coefficients
            indices, coefficients = self.design_fir_filter(filter_type, fs, stop_band_attenuation, fc, transition_band, fc2 )

            # Update the result text with filter coefficients
            self.result_text.delete("1.0", tk.END)
            for i, coef in zip(indices, coefficients):
                self.result_text.insert(tk.END, f"{i}: {coef:.6f}\n")

            # Save as indices and samples arrays
            self.indices_array = np.array(indices)
            self.samples_array = np.array(coefficients)

            # Store the filtered signal for later use (e.g., in convolution)
            self.filtered_signal = {'indices': self.indices_array, 'samples': self.samples_array}

            # Print the filtered signal in the result text box
            self.result_text.insert(tk.END, "\nFiltered Signal:\n")
            for idx, sample in zip(self.filtered_signal['indices'], self.filtered_signal['samples']):
                self.result_text.insert(tk.END, f"Index: {idx}, Sample: {sample:.6f}\n")

            if filter_type == 'Low pass':
                Compare_Signals("FIR Test cases/LPFCoefficients.txt", self.indices_array, self.samples_array)
            elif filter_type == 'High pass':
                Compare_Signals("FIR Test cases/HPFCoefficients.txt", self.indices_array, self.samples_array)
            elif filter_type == 'Band pass':
                 Compare_Signals("FIR Test cases/BPFCoefficients.txt", self.indices_array, self.samples_array)
            elif filter_type == 'Band stop':
                 Compare_Signals("FIR Test cases/BSFCoefficients.txt", self.indices_array, self.samples_array)

            return self.filtered_signal

        except ValueError as e:
             self.result_text.delete("1.0", tk.END)
             self.result_text.insert(tk.END, f"Error: {e}")

    def design_fir_filter(self, filter_type, fs, stop_band_attenuation, fc, transition_band, fc2=None):
        def window_function(stop_band_attenuation, n, N):
            if stop_band_attenuation <= 21:  # Rectangular
                return 1
            elif stop_band_attenuation <= 44:  # Hanning
                return 0.5 + (0.5 * np.cos((2 * np.pi * n) / N))
            elif stop_band_attenuation <= 53:  # Hamming
                return 0.54 + (0.46 * np.cos((2 * np.pi * n) / N))
            elif stop_band_attenuation <= 74:  # Blackman
                return 0.42 + (0.5 * np.cos(2 * np.pi * n / (N - 1))) + 0.08 * np.cos(4 * np.pi * n / (N - 1))

        def round_up_to_odd(number):
            rounded_number = math.ceil(number)
            if rounded_number % 2 == 0:
                rounded_number += 1
            return rounded_number

        delta_f = transition_band / fs
        if stop_band_attenuation <= 21:
            N = round_up_to_odd(0.9 / delta_f)
        elif stop_band_attenuation <= 44:
            N = round_up_to_odd(3.1 / delta_f)
        elif stop_band_attenuation <= 53:
            N = round_up_to_odd(3.3 / delta_f)
        elif stop_band_attenuation <= 74:
            N = round_up_to_odd(5.5 / delta_f)

        h = []
        indices = range(-math.floor(N / 2), math.floor(N / 2) + 1)

        if filter_type == 'Low pass':
            new_fc = fc + 0.5 * transition_band
            new_fc = new_fc / fs
            for n in indices:
                w_n = window_function(stop_band_attenuation, n, N)
                h_d = 2 * new_fc if n == 0 else 2 * new_fc * (np.sin(n * 2 * np.pi * new_fc) / (n * 2 * np.pi * new_fc))
                h.append(h_d * w_n)

        elif filter_type == 'High pass':
            new_fc = fc - 0.5 * transition_band
            new_fc /= fs
            for n in indices:
                w_n = window_function(stop_band_attenuation, n, N)
                h_d = 1 - 2 * new_fc if n == 0 else -2 * new_fc * (np.sin(n * 2 * np.pi * new_fc) / (n * 2 * np.pi * new_fc))
                h.append(h_d * w_n)

        elif filter_type == 'Band pass' and fc2 is not None:
            new_fc = fc - 0.5 * transition_band
            new_fc /= fs
            new_fc2 = fc2 + 0.5 * transition_band
            new_fc2 /= fs
            for n in indices:
                w_n = window_function(stop_band_attenuation, n, N)
                if n == 0:
                    h_d = 2 * (new_fc2 - new_fc)
                else:
                    h_d = 2 * new_fc2 * (np.sin(n * 2 * np.pi * new_fc2) / (n * 2 * np.pi * new_fc2)) - \
                          2 * new_fc * (np.sin(n * 2 * np.pi * new_fc) / (n * 2 * np.pi * new_fc))
                h.append(h_d * w_n)

        elif filter_type == 'Band stop' and fc2 is not None:
            new_fc = fc + 0.5 * transition_band
            new_fc /= fs
            new_fc2 = fc2 - 0.5 * transition_band
            new_fc2 /= fs
            for n in indices:
                w_n = window_function(stop_band_attenuation, n, N)
                if n == 0:
                    h_d = 1 - 2 * (new_fc2 - new_fc)
                else:
                    h_d = 2 * new_fc * (np.sin(n * 2 * np.pi * new_fc) / (n * 2 * np.pi * new_fc)) - \
                          2 * new_fc2 * (np.sin(n * 2 * np.pi * new_fc2) / (n * 2 * np.pi * new_fc2))
                h.append(h_d * w_n)

        return indices, h



    def resample_signal(self):
        try:
            L = int(self.resample_l_factor_entry.get())  
            M = int(self.resample_factor_entry.get())  
           

            if not hasattr(self, 'signals') or not self.signals:
                messagebox.showerror("Error", "No signals loaded.")
                return

            input_signal = self.signals[0]
            input_indices = input_signal['indices']
            input_samples = input_signal['samples']

            if M == 0 and L == 0:
                messagebox.showerror("Error", "Both M and L cannot be zero.")
                return
            #upsamplind
            if M == 0 and L != 0:
                upsampled_indices,upsampled_samples = self.upsample(input_indices,input_samples, L)

                indices ,coefficients =self.design_fir_filter('Low pass',8000,50,1500,500)
                self.indices_array = np.array(indices)
                self.samples_array = np.array(coefficients)

                self.filtered_signal = {'indices': self.indices_array, 'samples': self.samples_array}
                if len(self.signals) > 0:
                   self.signals[0] = {"indices": upsampled_indices, "samples": upsampled_samples}
                else:
                  # If signals is empty, add a new signal
                   self.signals.append({"indices": upsampled_indices, "samples": upsampled_samples})
                filtered_signal =self.convolution()
                self.display_signal({'indices': filtered_signal['indices'], 'samples': filtered_signal['samples']}, "Upsampled Signal ")
                Compare_Signals("FIR Test cases/Sampling_Up.txt",filtered_signal['indices'],filtered_signal['samples'])


            #downsamplind
            elif M != 0 and L == 0:
                indices ,coefficients =self.design_fir_filter('Low pass',8000,50,1500,500)
                self.indices_array = np.array(indices)
                self.samples_array = np.array(coefficients)

                self.filtered_signal = {'indices': self.indices_array, 'samples': self.samples_array}

                filtered_signal =self.convolution()
               
                downsampled_indices ,downsampled_samples= self.downsample(filtered_signal['indices'],filtered_signal['samples'], M)
                self.display_signal({'indices': downsampled_indices, 'samples': downsampled_samples}, "Downsampled Signal ")
                Compare_Signals("FIR Test cases/Sampling_Down.txt", downsampled_indices, downsampled_samples)


            elif M != 0 and L != 0:
                upsampled_indices,upsampled_samples = self.upsample(input_indices, input_samples,L)
                # filter
                indices ,coefficients =self.design_fir_filter('Low pass',8000,50,1500,500)
                self.indices_array = np.array(indices)
                self.samples_array = np.array(coefficients)

                self.filtered_signal = {'indices': self.indices_array, 'samples': self.samples_array}
                if len(self.signals) > 0:
                   self.signals[0] = {"indices": upsampled_indices, "samples": upsampled_samples}
                else:
                  # If signals is empty, add a new signal
                   self.signals.append({"indices": upsampled_indices, "samples": upsampled_samples})
                filtered_signal =self.convolution()
               
                # Downsampling
                downsampled_indices ,downsampled_samples= self.downsample(filtered_signal['indices'],filtered_signal['samples'], M)
                self.display_signal({'indices': downsampled_indices, 'samples': downsampled_samples}, "Sampling_Up_Down Signal ")
                Compare_Signals("FIR Test cases/Sampling_Up_Down.txt", downsampled_indices, downsampled_samples)

        except ValueError as e:
            messagebox.showerror("Error", f"Error during resampling: {e}")

    def upsample(self, indices,samples, L):

        start = indices[0]
        end = indices[0] + (len(indices) - 1) * L
        new_indices = list(range(start, end + 1))
        upsampled_signal = []
    
        for i in range(len(samples)):
        # Append the current signal value
          upsampled_signal.append(samples[i])
        
           # Insert L zeros after each signal value (except the last one)
          if i < len(samples) - 1:
            upsampled_signal.extend([0] * (L-1))
         
    
        return new_indices, upsampled_signal



    def downsample(self,indices, samples, M):
       
       downsampled_samples = samples[::M]  
       return indices[:len(downsampled_samples)], downsampled_samples

    def display_signal(self, signal, description):
        """
        Displays the signal in the result box.
        """
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"{description}:\n")
        for idx, sample in zip(signal['indices'], signal['samples']):
            self.result_text.insert(tk.END, f"Index: {idx}, Sample: {sample:.6f}\n")

