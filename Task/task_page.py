import tkinter as tk

class TaskPage(tk.Frame):
    def __init__(self, parent, task_name):
        super().__init__(parent, bg="#FFFFFF")
        self.task_name = task_name

        # Page title
        title_label = tk.Label(self, text=task_name, font=("Helvetica", 24, "bold"), bg="#FFFFFF")
        title_label.pack(pady=20)

        details_label = tk.Label(self, text=f"Details for {task_name} go here.", font=("Helvetica", 16), bg="#FFFFFF")
        details_label.pack(pady=10)

        check_button = tk.Checkbutton(self, text="Mark as Complete", font=("Helvetica", 14), bg="#FFFFFF")
        check_button.pack(pady=10)

        back_button = tk.Button(self, text="Back to Task List", command=self.back_to_list, font=("Helvetica", 14), bg="#3498DB", fg="white")
        back_button.pack(pady=20)

    def back_to_list(self):
        self.pack_forget()