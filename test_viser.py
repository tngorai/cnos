import viser
import time

server = viser.ViserServer()
dropdown = server.gui.add_dropdown("Level", ("0", "1", "2"), initial_value="2")
print("Dropdown added")

handle = server.scene.add_icosphere("/test", radius=1.0, position=(0,0,0), color=(255,0,0))
handle.remove()
print("Handle removed")
