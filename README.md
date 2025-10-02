# Digital Signal Processing (DSP) Application

Digital Signal Processing DSP Application

Overview
This DSP application provides tools for signal processing including signal sampling quantization Fourier Transform analysis and reconstruction The package is designed for ease of use with a GUI built using Tkinter

Features
Signal Sampling Load signals from a file and visualize sampled signals
Quantization Quantize signals using a specified number of levels or bits
Fourier Transform Analysis Compute and display the Discrete Fourier Transform DFT and its inverse IDFT
Signal Reconstruction Reconstruct signals from the IDFT
Customizable Parameters Adjust sampling frequency quantization levels and Fourier transform settings
User Interface Intuitive GUI for easy interaction with signal processing functions

Installation
1 Clone the repository
   git clone httpsgithubcomMaysoonHelmyDSPApplicationgit
2 Navigate to the project directory
   cd DSPApplication
3 Install dependencies
   pip install r requirementstxt

Usage
Run the main application
python mainpy

File Formats
Input Signals are loaded from text files and can include real and complex numbers
Output Processed signals are displayed in the GUI without being saved to new files

GUI Controls
Load Signal Open a file to import signal data
Set Sampling Frequency Define the frequency in Hz
Select Processing Option Choose from quantization DFT or IDFT
View Results Display processed signal outputs in tabular and graphical formats

Notes
Quantization follows the midpoint rule with special handling for boundary values
Fourier analysis is computed manually without built in FFT functions
Results are displayed with three significant figures
The project is actively maintained at GitHub Repository httpsgithubcomMaysoonHelmyDSPApplicationtreemain




