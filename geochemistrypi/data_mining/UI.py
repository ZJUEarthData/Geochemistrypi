import os
import tkinter
from tkinter import INSERT, Button, Label, Text, Tk, filedialog


class Geochemistrypi_GUI(object):
    def __init__(self, root):
        self.root = root
        self.name = None
        self.state = None
        self.font = ("Consolas", 10)
        self.root.title("Geochemistrypi")
        self.root.geometry("680x80")
        self.root.resizable(False, False)

    # Setup window
    def set_init_window(self):
        self.root.title("Geochemistrypi")
        self.root.geometry("750x100")
        self.root.resizable(False, False)
        # Label
        self.data_label = Label(self.root, text="File path", font=self.font)
        self.data_label.place(x=20, y=26, width=80, height=25)
        # Text
        self.data_Text = Text(self.root)
        self.test_Text = Text(self.root)
        self.data_Text.place(x=110, y=20, width=400, height=30)
        self.test_Text.pack_forget
        # Button
        self.run_button = Button(self.root, text="import", command=self.get_file_name, font=self.font)
        self.run_button.place(x=530, y=20, width=75, height=27)
        # self.stop_button = Button(self.root, text="confirm", command=self.stop, font=self.font)
        # self.stop_button.place(x=630, y=26, width=70, height=25)

    def get_file_name(self):
        # Acquisition path
        test = self.test_Text.get("1.0", "end")
        self.name = self.data_Text.get("1.0", "end")
        if self.name == test:
            name_path = filedialog.askopenfilename(title="Get data address", filetypes=[("Excel文件", "*.xlsx")])
            self.data_Text.insert(INSERT, name_path)
            if name_path == "":
                print("Error! You have not selected any files!")
            else:
                self.name = name_path
                self.state = 0
                self.stop()
        else:
            self.state = 1
            self.stop()

    def stop(self):
        if self.name == self.test_Text or self.state is None:
            tkinter.messagebox.showwarning("Warning", "You have not selected any files!")
            print("You have not selected any files!")
        elif ".xlsx" not in self.name:
            tkinter.messagebox.showwarning("Warning", "There may be a problem with the selected file format!")
            print("There may be a problem with the selected file format!\nYou need to select a file of type *.xlsx.")
        elif self.state == 1 and tkinter.messagebox.askyesno("Geochemistry π", "Import the file?"):
            print("The address of the file you want to obtain is:\n", self.name)
            self.root.quit()
            self.root.destroy()
        elif self.state == 0:
            print("The file address is successfully obtained:\n", self.name)
            self.root.quit()
            tkinter.messagebox.showinfo("Information", "The address of the file has been obtained successfully!")
            self.root.destroy()
        else:
            pass


def gui_data_input():
    # Instantiate a parent window
    init_window = Tk()
    # Set icon style
    current_dir = os.getcwd()
    path = os.path.dirname(os.path.dirname(current_dir)) + "\docs\Geochemistry π.ico"
    init_window.iconbitmap(path)
    # Instantiate the class
    data_gui = Geochemistrypi_GUI(init_window)
    # Set the default properties of the root window
    data_gui.set_init_window()
    # Keep window running
    init_window.mainloop()
    return data_gui.name


gui_data_input()
