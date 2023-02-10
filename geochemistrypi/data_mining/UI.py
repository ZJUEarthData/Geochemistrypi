from tkinter import *
from tkinter import filedialog

class Geochemistrypi_GUI():
    def __init__(self,root):
        self.root = root
        self.name = None

    # Setup window
    def set_init_window(self):
        self.root.title("Geochemistrypi")
        self.root.geometry('750x100')
        self.root.resizable(False, False)
        # Label
        self.data_label = Label(self.root, text="File path", font=("TimesNewRoman", 12))
        self.data_label.place(x=20, y=26, width=80, height=25)
        # Text
        self.data_Text = Text(self.root)
        self.test_Text = Text(self.root)
        self.data_Text.place(x=110, y=20, width=400, height=40)
        self.test_Text.pack_forget
        # Button
        self.run_button = Button(self.root, text="import", command=self.getNameFile)
        self.run_button.place(x=530, y=26, width=70, height=25)
        self.stop_button = Button(self.root, text="exit", command=self.stop)
        self.stop_button.place(x=630, y=26, width=70, height=25)

    def getNameFile(self):
        # Acquisition path
        test = self.test_Text.get("1.0", "end")
        self.name = self.data_Text.get("1.0", "end")
        if (self.name == test):
            name_path = filedialog.askopenfilename()
            self.name = name_path
            self.data_Text.insert(INSERT, self.name)
            if (self.name == ""):
                print("error!")
            print('Successfully obtain file:\n', self.name)
        else:
            print("The input content has been obtained:\n", self.name)

    def stop(self):
        self.root.quit()

def gui_data_input():
    # Instantiate a parent window
    init_window = Tk()
    data_gui = Geochemistrypi_GUI(init_window)
    # Set the default properties of the root window
    data_gui.set_init_window()
    # Keep window running
    init_window.mainloop()
    return data_gui.name