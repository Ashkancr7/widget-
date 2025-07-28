import FreeSimpleGUI as fg 

input_box1 = fg.InputText(tooltip="hay moon!", key="work")
button1 = fg.Button("add")
list_box = fg.Listbox(values="", key="works",
                      enable_events=True, size=[25,10])


window = fg.Window("moon",layout=[[list_box],[input_box1],[button1]])

window.read()
window.close()