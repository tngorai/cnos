# blenderproc run src/poses/create_template_poses.py
import blenderproc
import math
import numpy as np
import bpy
import bmesh
import math



def get_camera_positions_cone(center=(0.0, 0.0, 0.0),
                              num_layers=12,
                              layer_distance=85.0,
                              cone_radius=500.0,
                              min_height=50.0,
                              cameras_per_ring=36):
    # positions stores 3D coordinates (x,y,z) of each camera location.
    # angles stores the corresponding azimuth/elevation pairs.
    positions = []
    angles = []

    # at least 3 cameras per ring
    n_az = max(3, cameras_per_ring)
    cx, cy, cz = center
    
    total_height = min_height + max(0, num_layers - 1) * layer_distance

    for i in range(num_layers):
        # Height of this ring
        h = min_height + i * layer_distance

        if total_height > 0:
            r = cone_radius * (h / total_height)
        else:
            r = cone_radius

        # Skip rings that are too close to the apex (radius nearly zero)
        if r < 1e-6:
            continue

        # Place exactly 'n_az' points equally spaced around this circle
        for j in range(n_az):
            theta = 2 * math.pi * j / n_az

            x = r * math.cos(theta) + cx
            y = r * math.sin(theta) + cy
            z = h + cz

            positions.append((x, y, z))

            # Compute azimuth/elevation relative to center
            rel_x, rel_y, rel_z = x - cx, y - cy, z - cz
            az = math.atan2(rel_x, rel_y)
            el = math.atan2(rel_z, math.sqrt(rel_x * rel_x + rel_y * rel_y))
            angles.append((el, az))

    # Sort consistently (by elevation then azimuth)
    data = zip(angles, positions)
    data = sorted(data)
    positions = [p for _, p in data]
    angles = sorted(angles)

    return angles, positions




def get_camera_positions(nSubDiv):
    """
    * Construct an icosphere
    * subdived
    """

    bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 0), enter_editmode=True)
    # bpy.ops.export_mesh.ply(filepath='./sphere.ply')
    icos = bpy.context.object
    me = icos.data

    # -- cut away lower part
    bm = bmesh.from_edit_mesh(me)
    sel = [v for v in bm.verts if v.co[2] < 0]

    bmesh.ops.delete(bm, geom=sel, context="FACES")
    bmesh.update_edit_mesh(me)

    # -- subdivide and move new vertices out to the surface of the sphere
    #    nSubDiv = 3
    for i in range(nSubDiv):
        bpy.ops.mesh.subdivide()

        bm = bmesh.from_edit_mesh(me)
        for v in bm.verts:
            l = math.sqrt(v.co[0] ** 2 + v.co[1] ** 2 + v.co[2] ** 2)
            v.co[0] /= l
            v.co[1] /= l
            v.co[2] /= l
        bmesh.update_edit_mesh(me)

    # -- cut away zero elevation
    bm = bmesh.from_edit_mesh(me)
    sel = [v for v in bm.verts if v.co[2] <= 0]
    bmesh.ops.delete(bm, geom=sel, context="FACES")
    bmesh.update_edit_mesh(me)

    # convert vertex positions to az,el
    positions = []
    angles = []
    bm = bmesh.from_edit_mesh(me)
    for v in bm.verts: # look at this
        x = v.co[0]
        y = v.co[1]
        z = v.co[2]
        az = math.atan2(x, y)  # *180./math.pi
        el = math.atan2(z, math.sqrt(x**2 + y**2))  # *180./math.pi
        # positions.append((az,el))
        angles.append((el, az))
        positions.append((x, y, z))

    bpy.ops.object.editmode_toggle()

    # sort positions, first by az and el
    data = zip(angles, positions)
    positions = sorted(data)
    positions = [y for x, y in positions]
    angles = sorted(angles)
    return angles, positions



def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True))


def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0.0, 0.0, -1.0])
    # print warning when camera location is parallel to tmp
    norm = min(
        np.linalg.norm(cam_location - tmp, axis=-1),
        np.linalg.norm(cam_location + tmp, axis=-1),
    )
    if norm < 1e-3:
        print("Warning: camera location is parallel to tmp")
        tmp = np.array([0.0, -1.0, 0.0])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def convert_location_to_rotation(locations, center=(0.0, 0.0, 0.0)):
    obj_poses = np.zeros((len(locations), 4, 4))
    for idx, pt in enumerate(locations):
        obj_poses[idx] = look_at(pt, np.array(center))
    return obj_poses


def inverse_transform(poses):
    new_poses = np.zeros_like(poses)
    for idx_pose in range(len(poses)):
        rot = poses[idx_pose, :3, :3]
        t = poses[idx_pose, :3, 3]
        rot = np.transpose(rot)
        t = -np.matmul(rot, t)
        new_poses[idx_pose][3][3] = 1
        new_poses[idx_pose][:3, :3] = rot
        new_poses[idx_pose][:3, 3] = t
    return new_poses


import argparse
import sys

# Extract args after "--" if they exist, to play nice with blenderproc
try:
    idx = sys.argv.index("--")
    script_args = sys.argv[idx+1:]
except ValueError:
    script_args = sys.argv[1:]

parser = argparse.ArgumentParser()
parser.add_argument('--center_x', type=float, default=0.0)
parser.add_argument('--center_y', type=float, default=0.0)
parser.add_argument('--center_z', type=float, default=0.0)
parser.add_argument('--cameras_per_ring', type=int, default=36, help='Number of cameras per layer')
parser.add_argument('--num_layers', type=int, default=12, help='Total number of layers')
parser.add_argument('--layer_distance', type=float, default=85.0, help='Distance between consecutive layers')
parser.add_argument('--cone_radius', type=float, default=500.0, help='Radius at the top base of the cone')
parser.add_argument('--min_height', type=float, default=50.0, help='Minimum height from center to start rings')
args = parser.parse_args(script_args)

import os

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predefined_poses")

# Ensure directory exists just in case
os.makedirs(save_dir, exist_ok=True)

center_pt = (args.center_x, args.center_y, args.center_z)

for level in [0, 1, 2]:
    # Generate cone positions (saving them for all levels to make it easy)
    angles, positions = get_camera_positions_cone(
        center=center_pt,
        num_layers=args.num_layers,
        layer_distance=args.layer_distance,
        cone_radius=args.cone_radius,
        min_height=args.min_height,
        cameras_per_ring=args.cameras_per_ring
    )
    position_icosphere = np.asarray(positions)
    cam_poses = convert_location_to_rotation(position_icosphere, center=center_pt)
    np.save(f"{save_dir}/cam_poses_level{level}.npy", cam_poses)
    obj_poses = inverse_transform(cam_poses)
    np.save(f"{save_dir}/obj_poses_level{level}.npy", obj_poses)
    mat = np.load(f"{save_dir}/cam_poses_level{level}.npy")
    print(mat.shape)

print("Output saved to: " + save_dir)
