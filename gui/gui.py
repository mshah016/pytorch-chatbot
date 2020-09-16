# chatbot gui

# dependencies
from tkinter import * 

# create main chatbot window object 
main = Tk()

# assign properties to window object
main.title("Chat")
# assign size of window
main.geometry("400x500")
# disable resize quality
main.resizable(width = False, height = False)

# add toolbar to the window(i.e. File, Exit, etc)
toolbar = Menu(main) 

# add file menu
file_menu = Menu(main)

# add file commands
file_menu.add_command(label='New')
file_menu.add_command(label='Save')
file_menu.add_command(label='Quit')

# make file_menu a cascading menu in the toolbar
toolbar.add_cascade(label="File", menu=file_menu)

# add quit command in toolbar 
toolbar.add_command(label="Exit")

# configure the main window to contain the toolbar
main.config(menu=toolbar)


# create conversation window 
chatWindow = Text(main, bd=1, bg="black", width="50", height="8", font=("Dosis", 20), foreground="#00ffff")
chatWindow.place(x=6, y=6, width=370, height=385)

# create user's message window
messageWindow = Text(main, bd=0, bg="black",width="30", height="4", font=("Dosis", 20), foreground="#00ffff")
# placement of window
messageWindow.place(x=128, y=400, height=88, width=260)

# create scroll bar in the conversation window
scrollbar = Scrollbar(main, command=chatWindow.yview, cursor='star')
# placement of scrollbar 
scrollbar.place(x=375, y=5, height= 385)

# create send button
button = Button(main, text="Send", width="12", height=5,
                        bd=0, bg="#0080ff",
                        activebackground="#00bfff",
                        foreground="#ffffff",
                        font=("Dosis", 15))
# placement of button, next to messageWindow
button.place(x=6, y=400, height=88)

# run mainloop()
main.mainloop()