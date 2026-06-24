import numpy as np
import viser
import viser.transforms as tf
import time
import subprocess
import os

# ====== CONFIGURATION ======
NPY_DIR = "/home/aryan/Documents/cnos/src/poses/predefined_poses"
LEVEL = 2
DEFAULT_LEVEL = "2"
ARROW_SCALE = 300.0  # reference scale for arrow thickness
DEFAULT_SPHERE_RADIUS = 20.0  # Initial size of red spheres


# ===========================

def load_camera_poses(level):
    file_path = f"{NPY_DIR}/cam_poses_level{level}.npy"
    try:
        poses = np.load(file_path)
        print(f"Loaded {len(poses)} poses from {file_path}")
        return poses
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None


def main():
    poses = load_camera_poses(int(DEFAULT_LEVEL))
    if poses is None:
        poses = []

    # Start the viser server
    server = viser.ViserServer()
    print("Viser server started. Open http://localhost:8080 in your browser.")

    # ---- Set initial camera view ----
    server.initial_camera.position = (2000.0, 1500.0, 1000.0)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)
    server.initial_camera.far = 10000.0

    # ---- Reference: green sphere at origin ----
    server.scene.add_icosphere(
        "/origin_sphere",
        radius=50.0,
        position=(0, 0, 0),
        color=(0, 255, 0),
    )

    # ---- Interactive GUI Functions ----
    scene_handles = []

    def update_scene():
        """Redraws the camera spheres and places clean, non-overlapping directional arrows."""
        current_radius = sphere_slider.value

        for handle in scene_handles:
            handle.remove()
        scene_handles.clear()

        for i, pose in enumerate(poses):
            pos = pose[:3, 3]
            rot = pose[:3, :3]
            wxyz = tf.SO3.from_matrix(rot).wxyz

            # 1. Red sphere at camera position
            h_sphere = server.scene.add_icosphere(
                f"/cameras/sphere_{i}",
                radius=current_radius,
                position=pos,
                color=(255, 0, 0),
            )
            scene_handles.append(h_sphere)

            # 2. DYNAMIC BLUE ARROW: Starts exactly at the surface of the sphere (Z = current_radius)
            arrow_length = 50.0
            arrow_points = np.array([[[0.0, 0.0, current_radius], [0.0, 0.0, current_radius + arrow_length]]],
                                    dtype=np.float32)

            h_arrow = server.scene.add_arrows(
                f"/frames/arrow_{i}",
                points=arrow_points,
                colors=(0, 0, 255),  # Blue color
                shaft_radius=ARROW_SCALE * 0.01,  # Skinny shaft
                head_radius=ARROW_SCALE * 0.025,  # Proportionate head radius
                head_length=arrow_length * 0.4,  # Head cone takes up 40% of the short arrow
                position=pos,
                wxyz=wxyz,
            )
            scene_handles.append(h_arrow)

    # Add a control panel GUI folder in the sidebar
    with server.gui.add_folder("Controls"):
        level_dropdown = server.gui.add_dropdown(
            label="Pose Level",
            options=["0", "1", "2"],
            initial_value=DEFAULT_LEVEL,
        )
        sphere_slider = server.gui.add_slider(
            label="Sphere Size",
            min=5.0,
            max=300.0,
            step=5.0,
            initial_value=DEFAULT_SPHERE_RADIUS,
        )
        
    with server.gui.add_folder("Generator"):
        center_x_input = server.gui.add_number("Center X", initial_value=0.0)
        center_y_input = server.gui.add_number("Center Y", initial_value=0.0)
        center_z_input = server.gui.add_number("Center Z", initial_value=0.0)
        
        cameras_per_ring_input = server.gui.add_number(
            label="Cameras per Layer",
            initial_value=36,
        )
        num_layers_input = server.gui.add_number(
            label="Num Layers",
            initial_value=12,
        )
        layer_distance_input = server.gui.add_number(
            label="Distance between layers",
            initial_value=85.0,
        )
        cone_radius_input = server.gui.add_number(
            label="Cone Radius",
            initial_value=500.0,
        )
        min_height_input = server.gui.add_number(
            label="Min Height",
            initial_value=50.0,
        )
        generate_btn = server.gui.add_button("Generate Custom Poses")
        status_text = server.gui.add_markdown("Status: Ready")

    # Link the slider event to our update function
    @sphere_slider.on_update
    def _(_):
        update_scene()
        
    @level_dropdown.on_update
    def _(_):
        nonlocal poses
        new_poses = load_camera_poses(int(level_dropdown.value))
        if new_poses is not None:
            poses = new_poses
            update_scene()
            print(f"Visualized {len(poses)} cameras for level {level_dropdown.value}.")

    @generate_btn.on_click
    def _(_):
        status_text.content = "**Status: Generating... Please wait.**\n*(This uses BlenderProc and may take a moment)*"
        
        # Give the GUI a tiny fraction of a second to broadcast the markdown change
        time.sleep(0.1)
        
        val_cx = float(center_x_input.value)
        val_cy = float(center_y_input.value)
        val_cz = float(center_z_input.value)
        val_cpr = int(cameras_per_ring_input.value)
        val_layers = int(num_layers_input.value)
        val_dist = float(layer_distance_input.value)
        val_radius = float(cone_radius_input.value)
        val_min_height = float(min_height_input.value)
        
        try:
            # Use absolute path to ensure blenderproc can find the script regardless of current working directory
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "create_template_poses.py"))
            
            subprocess.run(
                ["blenderproc", "run", script_path, "--", 
                 "--center_x", str(val_cx),
                 "--center_y", str(val_cy),
                 "--center_z", str(val_cz),
                 "--cameras_per_ring", str(val_cpr),
                 "--num_layers", str(val_layers),
                 "--layer_distance", str(val_dist),
                 "--cone_radius", str(val_radius),
                 "--min_height", str(val_min_height)],
                check=True
            )
            
            status_text.content = "**Status: Loading new poses...**"
            nonlocal poses
            new_poses = load_camera_poses(int(level_dropdown.value))
            if new_poses is not None:
                poses = new_poses
                update_scene()
                
            status_text.content = "Status: Ready"
            
        except subprocess.CalledProcessError as e:
            status_text.content = f"**Status: Error generating!**\n`{e}`"
        except Exception as e:
            status_text.content = f"**Status: Error!**\n`{str(e)}`"

    # Draw the initial spheres and arrows on startup
    update_scene()

    print(f"Visualized {len(poses)} cameras with interactive controls.")
    print("Press Ctrl+C to stop.")

    # Keep server alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down server...")


if __name__ == "__main__":
    main()