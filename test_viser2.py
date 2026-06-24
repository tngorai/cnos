import viser
server = viser.ViserServer()
num = server.gui.add_number("Cameras/Ring", initial_value=36)
btn = server.gui.add_button("Generate Poses")
print("Controls added")
