import tkinter as tk
from tkinter import ttk
from task_page1 import TaskPage1
from task_page2 import TaskPage2
from task_page3 import TaskPage3
from task4 import TaskPage4
from task_page6 import TaskPage6
from task_page7 import Task7
from Project import EOGTaskPage

class DSPTasksPage(tk.Frame):
    def __init__(self, parent, task_name):
        super().__init__(parent, bg="#FFFFFF")
        label = tk.Label(self, text="DSP TASKS", font=("Helvetica", 50, "bold"), bg="#FFFFFF")
        label.pack(expand=True)

class TaskApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("DSP Tasks")
        self.geometry("1900x900")
        self.configure(bg="#F8F9FA")

        # Create a main frame
        main_frame = tk.Frame(self, bg="#F8F9FA")
        main_frame.pack(fill="both", expand=True)

        # Create a frame for the task list (left side)
        self.task_list_frame = tk.Frame(main_frame, bg="#D9E3E0", width=450)
        self.task_list_frame.pack(side="left", fill="y")

        # Create a frame for task details (right side)
        self.task_detail_frame = tk.Frame(main_frame, bg="#FFFFFF")
        self.task_detail_frame.pack(side="right", fill="both", expand=True)

        # Store frames (task pages) in a dictionary
        self.frames = {}

        # List of tasks
        tasks = [ "DSP_TASKS", "Load & Generate Signal","Arithmetic Operations","Quantization", "Frequency Domain", "Convolution & Correlation", "FIR Filter and Resampling","EOG Right and Left"]

        # Create task list on the left side
        self.create_task_list(tasks)

        # Create a page for each task and store it in the dictionary
        for task in tasks:
            if task == "DSP_TASKS":
                page = DSPTasksPage(self.task_detail_frame, task)
            elif task == "Load & Generate Signal":
                page = TaskPage1(self.task_detail_frame, task)
            elif task == "Arithmetic Operations":
                page = TaskPage2(self.task_detail_frame, task)
            elif task == "Quantization":
                page = TaskPage3(self.task_detail_frame, task)
            elif task == "Frequency Domain":
                page = TaskPage4(self.task_detail_frame, task)
            elif task == "Convolution & Correlation":
                page = TaskPage6(self.task_detail_frame, task)
            elif task == "FIR Filter and Resampling":
                page = Task7(self.task_detail_frame)
            elif task == "EOG Right and Left":
                page = EOGTaskPage(self.task_detail_frame, task)  # Create EOG task page

            self.frames[task] = page

        # Show the DSP Tasks page initially
        self.show_task_details("DSP_TASKS")

        # Set the protocol for closing the window
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_task_list(self, tasks):
        label = tk.Label(self.task_list_frame, text="Tasks", font=("Helvetica", 20, "bold"), bg="#D9E3E0")
        label.pack(pady=20)

        # Create a button for each task
        for task in tasks:
            button = ttk.Button(self.task_list_frame, text=task, command=lambda t=task: self.show_task_details(t), width=20)
            button.pack(pady=10, padx=10, fill='x')

    def show_task_details(self, task_name):
        # Hide all task frames
        for frame in self.frames.values():
            frame.pack_forget()

        # Show the selected task frame
        if task_name in self.frames:
            self.frames[task_name].pack(fill="both", expand=True)

    def on_closing(self):
        self.destroy()

if __name__ == "__main__":
    app = TaskApp()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', font=('Helvetica', 14), padding=10, background='#3498DB', foreground='#FFFFFF')
    app.mainloop()
