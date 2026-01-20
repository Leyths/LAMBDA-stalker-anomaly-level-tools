"""
Factory class for creating spawn point mesh templates.

This module contains all the mesh generation code for different spawn types,
separated from the main geometry manager for better organization.
"""
import numpy as np
import open3d as o3d


class SpawnMeshFactory:
    """Factory for creating spawn point mesh templates."""

    # Default spawn size
    SPAWN_POINT_SIZE = 0.5

    # Spawn colors
    SPAWN_COLOR = [0, 0.8, 0]  # Green (default)
    LEVEL_CHANGER_COLOR = [1.0, 0.5, 0.0]  # Orange
    LIGHTS_COLOR = [1.0, 1.0, 0.0]  # Yellow
    PHYSIC_COLOR = [1.0, 0.3, 0.0]  # Red-orange
    SPACE_RESTRICTOR_COLOR = [0.2, 0.8, 0.2]  # Light green
    AMMO_COLOR = [0.8, 0.6, 0.2]  # Brass/bronze
    INVENTORY_BOX_COLOR = [1.0, 0.84, 0.0]  # Gold
    ZONE_COLOR = [0.2, 0.6, 1.0]  # Bright blue
    WEAPON_COLOR = [0.3, 0.3, 0.35]  # Dark gunmetal grey
    DETECTOR_COLOR = [0.2, 0.2, 0.2]  # Dark grey body
    DETECTOR_SCREEN_COLOR = [0.1, 0.4, 0.1]  # Dark green screen
    DETECTOR_BUTTON_COLORS = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.8, 0.8, 0.2]]
    SMART_COVER_COLOR = [0.6, 0.5, 0.35]  # Sandy brown for sandbags
    CAMPFIRE_WOOD_COLOR = [0.4, 0.25, 0.1]  # Dark brown for logs
    CAMPFIRE_FLAME_COLOR = [1.0, 0.5, 0.0]  # Orange flames
    CAMPFIRE_EMBER_COLOR = [1.0, 0.2, 0.0]  # Red-orange embers
    LIGHTBULB_METAL_COLOR = [0.7, 0.7, 0.75]  # Silver/metal for bulb base
    BOX_WOOD_COLOR = [0.55, 0.35, 0.15]  # Medium brown wood
    MUG_COLOR = [0.85, 0.85, 0.8]  # Off-white ceramic
    PLATE_COLOR = [0.9, 0.9, 0.85]  # Light cream ceramic
    BOTTLE_COLOR = [0.3, 0.5, 0.3]  # Green glass
    LADDER_COLOR = [0.5, 0.35, 0.2]  # Wood brown

    def __init__(self, spawn_point_size: float = 0.5):
        """Initialize factory with configurable spawn point size."""
        self.spawn_point_size = spawn_point_size

    def create_default_spawn_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create default green box for regular spawns."""
        size = self.spawn_point_size
        box = o3d.geometry.TriangleMesh.create_box(
            width=size, height=size, depth=size
        )
        # Center the box on origin
        box.translate(np.array([-size / 2, -size / 2, -size / 2]))
        box.paint_uniform_color(self.SPAWN_COLOR)
        return box

    def create_level_changer_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create large orange sphere for level changers."""
        # 1.5x larger than default size
        radius = self.spawn_point_size * 1.5 / 2
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color(self.LEVEL_CHANGER_COLOR)
        return sphere

    def create_lightbulb_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create lightbulb shape with yellow glass bulb and metal base."""
        size = self.spawn_point_size

        # Bulb part (sphere) - the glass part (yellow)
        bulb_radius = size * 0.35
        bulb = o3d.geometry.TriangleMesh.create_sphere(radius=bulb_radius)
        bulb.translate([0, size * 0.15, 0])
        bulb.paint_uniform_color(self.LIGHTS_COLOR)

        # Screw base (cylinder) - metal color
        base_radius = bulb_radius * 0.6
        base_height = size * 0.25
        base = o3d.geometry.TriangleMesh.create_cylinder(radius=base_radius, height=base_height)
        base.translate([0, -size * 0.1, 0])
        base.paint_uniform_color(self.LIGHTBULB_METAL_COLOR)

        # Bottom tip (cone) - metal color
        tip_radius = base_radius * 0.4
        tip_height = size * 0.15
        tip = o3d.geometry.TriangleMesh.create_cone(radius=tip_radius, height=tip_height)
        # Flip the cone so it points down
        tip.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0]), center=[0, 0, 0])
        tip.translate([0, -size * 0.3, 0])
        tip.paint_uniform_color(self.LIGHTBULB_METAL_COLOR)

        # Combine (colors preserved since each part is painted before combining)
        lightbulb = bulb + base + tip
        return lightbulb

    def create_explosion_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create spikey explosion shape for physics objects."""
        # 40% bigger than original
        size = self.spawn_point_size * 0.6 * 1.4

        # Start with icosahedron
        mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius=size * 0.5)

        # Get vertices and make them spikier by scaling outward
        vertices = np.asarray(mesh.vertices)

        # Add spikes by duplicating vertices and pushing them outward
        spike_mesh = o3d.geometry.TriangleMesh()

        # Create spikes from center to each vertex
        spike_length = size * 0.8

        for vertex in vertices:
            direction = vertex / np.linalg.norm(vertex)

            # Create a thin cone for each spike
            spike = o3d.geometry.TriangleMesh.create_cone(radius=size * 0.1, height=spike_length)

            # Rotate cone to point in the direction of the vertex
            # Default cone points in +Y direction
            default_dir = np.array([0, 1, 0])
            rotation_axis = np.cross(default_dir, direction)
            if np.linalg.norm(rotation_axis) > 0.001:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(default_dir, direction), -1, 1))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                spike.rotate(rotation_matrix, center=[0, 0, 0])

            spike_mesh += spike

        # Add a small central sphere
        core = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.25)
        spike_mesh += core

        spike_mesh.paint_uniform_color(self.PHYSIC_COLOR)
        return spike_mesh

    def create_space_restrictor_marker_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create small green sphere marker for space restrictor spawns."""
        radius = self.spawn_point_size * 0.3
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color(self.SPACE_RESTRICTOR_COLOR)
        return sphere

    def create_ammo_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a small pile of bullet/shell-like cylinders in brass color."""
        size = self.spawn_point_size * 0.6
        combined = o3d.geometry.TriangleMesh()

        # Create several bullet-shaped cylinders arranged in a pile
        bullet_radius = size * 0.08
        bullet_height = size * 0.35

        # Base layer - 3 bullets lying flat
        positions = [
            ([0, 0, 0], [np.pi / 2, 0, 0]),                    # Center
            ([size * 0.15, 0, size * 0.1], [np.pi / 2, 0, 0.3]),   # Right
            ([-size * 0.12, 0, size * 0.08], [np.pi / 2, 0, -0.2]),  # Left
        ]

        # Second layer - 2 bullets on top
        positions += [
            ([size * 0.05, size * 0.08, size * 0.05], [np.pi / 2, 0, 0.5]),
            ([-size * 0.08, size * 0.08, size * 0.02], [np.pi / 2, 0, -0.4]),
        ]

        for pos, rot in positions:
            # Create bullet body (cylinder)
            bullet = o3d.geometry.TriangleMesh.create_cylinder(
                radius=bullet_radius, height=bullet_height
            )
            # Add rounded tip (small sphere at one end)
            tip = o3d.geometry.TriangleMesh.create_sphere(radius=bullet_radius * 0.9)
            tip.translate([0, bullet_height / 2, 0])
            bullet += tip

            # Rotate and position
            bullet.rotate(
                o3d.geometry.get_rotation_matrix_from_xyz(rot),
                center=[0, 0, 0]
            )
            bullet.translate(pos)
            combined += bullet

        combined.paint_uniform_color(self.AMMO_COLOR)
        return combined

    def create_inventory_box_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a rectangular chest/crate shape in gold."""
        size = self.spawn_point_size

        # Main body - rectangular box (wider than tall, like a chest)
        width = size * 1.0
        height = size * 0.6
        depth = size * 0.7

        box = o3d.geometry.TriangleMesh.create_box(
            width=width, height=height, depth=depth
        )
        # Center on origin
        box.translate([-width / 2, -height / 2, -depth / 2])

        # Add a lid ridge on top (thin box)
        lid_height = height * 0.15
        lid = o3d.geometry.TriangleMesh.create_box(
            width=width * 1.05, height=lid_height, depth=depth * 1.05
        )
        lid.translate([
            -width * 1.05 / 2,
            height / 2 - lid_height / 2,
            -depth * 1.05 / 2
        ])

        chest = box + lid
        chest.paint_uniform_color(self.INVENTORY_BOX_COLOR)
        return chest

    def create_zone_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a high-fidelity lightning bolt shape in bright blue."""
        size = self.spawn_point_size * 1.5

        # Define lightning bolt vertices as a 2D polygon, then extrude
        thickness = size * 0.08  # Depth of the bolt

        # Lightning bolt profile points (x, y) - a proper jagged bolt shape
        bolt_points = [
            # Top point
            (0.0, 0.5),
            # Right edge going down
            (0.15, 0.5),
            (0.05, 0.15),
            (0.2, 0.15),
            (0.0, -0.15),
            (0.12, -0.15),
            # Bottom point (the tip)
            (-0.05, -0.5),
            # Left edge going up
            (0.0, -0.2),
            (-0.15, -0.2),
            (0.0, 0.1),
            (-0.12, 0.1),
            (-0.05, 0.5),
        ]

        # Scale points by size
        bolt_points = [(x * size, y * size) for x, y in bolt_points]

        # Create front and back faces using triangulation
        combined = o3d.geometry.TriangleMesh()

        # Create vertices for front and back faces
        vertices = []
        # Front face (z = thickness/2)
        for x, y in bolt_points:
            vertices.append([x, y, thickness / 2])
        # Back face (z = -thickness/2)
        for x, y in bolt_points:
            vertices.append([x, y, -thickness / 2])

        n = len(bolt_points)

        # Calculate centroid
        cx = sum(p[0] for p in bolt_points) / n
        cy = sum(p[1] for p in bolt_points) / n

        # Add center vertices for front and back
        vertices.append([cx, cy, thickness / 2])   # Front center: index 2*n
        vertices.append([cx, cy, -thickness / 2])  # Back center: index 2*n + 1

        triangles = []

        # Front face triangles (fan from center)
        front_center = 2 * n
        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([front_center, i, next_i])

        # Back face triangles (fan from center, reversed winding)
        back_center = 2 * n + 1
        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([back_center, n + next_i, n + i])

        # Side faces (connect front and back edges)
        for i in range(n):
            next_i = (i + 1) % n
            # Two triangles per side quad
            triangles.append([i, n + i, next_i])
            triangles.append([next_i, n + i, n + next_i])

        combined.vertices = o3d.utility.Vector3dVector(vertices)
        combined.triangles = o3d.utility.Vector3iVector(triangles)
        combined.compute_vertex_normals()
        combined.paint_uniform_color(self.ZONE_COLOR)
        return combined

    def create_weapon_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a small pistol shape in gunmetal grey."""
        size = self.spawn_point_size * 0.8
        combined = o3d.geometry.TriangleMesh()

        # Barrel (horizontal cylinder)
        barrel_length = size * 0.5
        barrel_radius = size * 0.06
        barrel = o3d.geometry.TriangleMesh.create_cylinder(
            radius=barrel_radius, height=barrel_length
        )
        barrel.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([0, 0, np.pi / 2]),
            center=[0, 0, 0]
        )
        barrel.translate([size * 0.15, size * 0.1, 0])
        combined += barrel

        # Slide (box on top of barrel)
        slide_length = size * 0.45
        slide_height = size * 0.12
        slide_width = size * 0.1
        slide = o3d.geometry.TriangleMesh.create_box(
            width=slide_length, height=slide_height, depth=slide_width
        )
        slide.translate([-slide_length / 2 + size * 0.1, size * 0.05, -slide_width / 2])
        combined += slide

        # Frame/receiver (box connecting slide to grip)
        frame_length = size * 0.3
        frame_height = size * 0.08
        frame_width = size * 0.09
        frame = o3d.geometry.TriangleMesh.create_box(
            width=frame_length, height=frame_height, depth=frame_width
        )
        frame.translate([-frame_length / 2, -size * 0.03, -frame_width / 2])
        combined += frame

        # Grip (angled box going down)
        grip_length = size * 0.12
        grip_height = size * 0.25
        grip_width = size * 0.08
        grip = o3d.geometry.TriangleMesh.create_box(
            width=grip_length, height=grip_height, depth=grip_width
        )
        grip.translate([-grip_length / 2, -grip_height / 2, -grip_width / 2])
        grip.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0.2]),
            center=[0, 0, 0]
        )
        grip.translate([-size * 0.08, -size * 0.15, 0])
        combined += grip

        # Trigger guard (small box)
        guard = o3d.geometry.TriangleMesh.create_box(
            width=size * 0.08, height=size * 0.02, depth=size * 0.06
        )
        guard.translate([size * 0.02, -size * 0.1, -size * 0.03])
        combined += guard

        combined.paint_uniform_color(self.WEAPON_COLOR)
        return combined

    def create_detector_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a detector device (TV remote style) with colored buttons."""
        size = self.spawn_point_size * 0.7
        combined = o3d.geometry.TriangleMesh()

        # Main body (rectangular)
        body_width = size * 0.25
        body_height = size * 0.6
        body_depth = size * 0.1
        body = o3d.geometry.TriangleMesh.create_box(
            width=body_width, height=body_height, depth=body_depth
        )
        body.translate([-body_width / 2, -body_height / 2, -body_depth / 2])
        body.paint_uniform_color(self.DETECTOR_COLOR)
        combined += body

        # Screen area (darker green rectangle on front)
        screen_width = body_width * 0.7
        screen_height = body_height * 0.35
        screen_depth = size * 0.01
        screen = o3d.geometry.TriangleMesh.create_box(
            width=screen_width, height=screen_height, depth=screen_depth
        )
        screen.translate([-screen_width / 2, body_height * 0.1, body_depth / 2])
        screen.paint_uniform_color(self.DETECTOR_SCREEN_COLOR)
        combined += screen

        # Buttons (small colored spheres/boxes below screen)
        button_size = size * 0.04
        button_y = -body_height * 0.15
        button_z = body_depth / 2 + button_size / 2

        button_positions = [
            (-body_width * 0.2, button_y, button_z),
            (0, button_y, button_z),
            (body_width * 0.2, button_y, button_z),
        ]

        for i, (bx, by, bz) in enumerate(button_positions):
            button = o3d.geometry.TriangleMesh.create_sphere(radius=button_size)
            button.translate([bx, by, bz])
            button.paint_uniform_color(self.DETECTOR_BUTTON_COLORS[i])
            combined += button

        # Second row of buttons
        button_y2 = -body_height * 0.3
        for i in range(2):
            bx = -body_width * 0.1 + i * body_width * 0.2
            button = o3d.geometry.TriangleMesh.create_box(
                width=button_size * 1.5, height=button_size, depth=button_size * 0.5
            )
            button.translate([bx - button_size * 0.75, button_y2, body_depth / 2])
            button.paint_uniform_color([0.5, 0.5, 0.5])  # Grey buttons
            combined += button

        # Antenna nub at top
        antenna = o3d.geometry.TriangleMesh.create_cylinder(
            radius=size * 0.02, height=size * 0.08
        )
        antenna.translate([0, body_height / 2 + size * 0.04, 0])
        antenna.paint_uniform_color(self.DETECTOR_COLOR)
        combined += antenna

        return combined

    def create_smart_cover_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a sandbag barricade shape."""
        size = self.spawn_point_size * 1.2
        combined = o3d.geometry.TriangleMesh()

        # Individual sandbag dimensions
        bag_width = size * 0.35
        bag_height = size * 0.18
        bag_depth = size * 0.18

        def create_sandbag(width, height, depth):
            """Create a single sandbag (rounded box shape using a squashed sphere)."""
            bag = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=8)
            vertices = np.asarray(bag.vertices)
            vertices[:, 0] *= width / 2   # X - width
            vertices[:, 1] *= height / 2  # Y - height
            vertices[:, 2] *= depth / 2   # Z - depth
            bag.vertices = o3d.utility.Vector3dVector(vertices)
            return bag

        # Bottom row - 3 sandbags side by side
        for i in range(3):
            bag = create_sandbag(bag_width, bag_height, bag_depth)
            x_offset = (i - 1) * bag_width * 0.9  # Slightly overlapping
            bag.translate([x_offset, bag_height / 2, 0])
            combined += bag

        # Second row - 2 sandbags staggered on top
        for i in range(2):
            bag = create_sandbag(bag_width, bag_height, bag_depth)
            x_offset = (i - 0.5) * bag_width * 0.9
            bag.translate([x_offset, bag_height * 1.4, 0])
            combined += bag

        # Top row - 1 sandbag on the very top (center)
        top_bag = create_sandbag(bag_width, bag_height, bag_depth)
        top_bag.translate([0, bag_height * 2.3, 0])
        combined += top_bag

        # Add a second layer behind for depth
        for i in range(2):
            bag = create_sandbag(bag_width * 0.9, bag_height, bag_depth)
            x_offset = (i - 0.5) * bag_width * 0.85
            bag.translate([x_offset, bag_height / 2, -bag_depth * 0.7])
            combined += bag

        combined.paint_uniform_color(self.SMART_COVER_COLOR)
        return combined

    def create_campfire_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a campfire with logs arranged in a circle and flames."""
        size = self.spawn_point_size * 1.7  # 70% bigger
        combined = o3d.geometry.TriangleMesh()

        # Log dimensions
        log_radius = size * 0.06
        log_length = size * 0.4

        # Create logs arranged in a tepee/crossed pattern
        num_logs = 5
        for i in range(num_logs):
            log = o3d.geometry.TriangleMesh.create_cylinder(
                radius=log_radius, height=log_length
            )
            # Tilt the log inward
            angle_around = (2 * np.pi * i) / num_logs
            tilt_angle = 0.4  # Tilt inward

            # Rotate to tilt
            log.rotate(
                o3d.geometry.get_rotation_matrix_from_xyz([tilt_angle, 0, 0]),
                center=[0, 0, 0]
            )
            # Rotate around Y axis to position around circle
            log.rotate(
                o3d.geometry.get_rotation_matrix_from_xyz([0, angle_around, 0]),
                center=[0, 0, 0]
            )
            # Move outward from center and up
            log.translate([
                np.sin(angle_around) * size * 0.12,
                log_length * 0.3,
                np.cos(angle_around) * size * 0.12
            ])
            log.paint_uniform_color(self.CAMPFIRE_WOOD_COLOR)
            combined += log

        # Add some horizontal logs at the base
        for i in range(3):
            base_log = o3d.geometry.TriangleMesh.create_cylinder(
                radius=log_radius * 0.8, height=log_length * 0.7
            )
            base_log.rotate(
                o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, i * np.pi / 3]),
                center=[0, 0, 0]
            )
            base_log.translate([0, log_radius, 0])
            base_log.paint_uniform_color(self.CAMPFIRE_WOOD_COLOR)
            combined += base_log

        # Create flames (cones pointing upward along Y axis)
        flame1 = o3d.geometry.TriangleMesh.create_cone(
            radius=size * 0.1, height=size * 0.35
        )
        flame1.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2, 0, 0]),
            center=[0, 0, 0]
        )
        flame1.translate([0, size * 0.3, 0])
        flame1.paint_uniform_color(self.CAMPFIRE_FLAME_COLOR)
        combined += flame1

        # Smaller side flames
        flame_positions = [
            (size * 0.06, 0, size * 0.04),
            (-size * 0.05, 0, size * 0.05),
            (size * 0.03, 0, -size * 0.06),
        ]
        for fx, fy, fz in flame_positions:
            flame = o3d.geometry.TriangleMesh.create_cone(
                radius=size * 0.06, height=size * 0.22
            )
            flame.rotate(
                o3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2, 0, 0]),
                center=[0, 0, 0]
            )
            flame.translate([fx, size * 0.2 + fy, fz])
            flame.paint_uniform_color(self.CAMPFIRE_FLAME_COLOR)
            combined += flame

        # Embers at the base (small spheres)
        for i in range(6):
            ember_angle = (2 * np.pi * i) / 6
            ember = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.03)
            ember.translate([
                np.sin(ember_angle) * size * 0.08,
                size * 0.02,
                np.cos(ember_angle) * size * 0.08
            ])
            ember.paint_uniform_color(self.CAMPFIRE_EMBER_COLOR)
            combined += ember

        return combined

    def create_wooden_box_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a small wooden box/crate shape."""
        size = self.spawn_point_size * 0.7

        # Main box body
        box = o3d.geometry.TriangleMesh.create_box(
            width=size, height=size * 0.7, depth=size
        )
        # Center on origin
        box.translate([-size / 2, -size * 0.35, -size / 2])
        box.paint_uniform_color(self.BOX_WOOD_COLOR)

        # Add lid/top planks (thin boxes on top for detail)
        plank_height = size * 0.05
        plank_width = size * 0.9
        plank_depth = size * 0.2

        combined = box
        for i in range(3):
            plank = o3d.geometry.TriangleMesh.create_box(
                width=plank_width, height=plank_height, depth=plank_depth
            )
            z_offset = -size * 0.35 + i * (plank_depth + size * 0.05)
            plank.translate([-plank_width / 2, size * 0.35, z_offset])
            plank.paint_uniform_color(self.BOX_WOOD_COLOR)
            combined += plank

        return combined

    def create_mug_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a small mug/cup shape."""
        size = self.spawn_point_size * 0.5

        # Mug body (cylinder) - created along Z, rotate to Y-up
        body_radius = size * 0.3
        body_height = size * 0.5
        body = o3d.geometry.TriangleMesh.create_cylinder(
            radius=body_radius, height=body_height
        )
        body.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0]),
            center=[0, 0, 0]
        )
        body.translate([0, body_height / 2, 0])
        body.paint_uniform_color(self.MUG_COLOR)

        # Handle (arc made from small spheres for simplicity)
        handle_segments = 6
        handle_radius = size * 0.12
        segment_radius = size * 0.035

        combined = body
        for i in range(handle_segments):
            angle = np.pi * i / (handle_segments - 1) - np.pi / 2
            segment = o3d.geometry.TriangleMesh.create_sphere(radius=segment_radius)
            # Position along the handle arc (on the side of the mug)
            x = body_radius + handle_radius * np.cos(angle)
            y = body_height * 0.5 + handle_radius * np.sin(angle)
            segment.translate([x, y, 0])
            segment.paint_uniform_color(self.MUG_COLOR)
            combined += segment

        return combined

    def create_plate_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a small plate/dish shape."""
        size = self.spawn_point_size * 0.6

        # Plate base (flat cylinder) - rotate to be horizontal
        plate_radius = size * 0.4
        plate_height = size * 0.05
        plate = o3d.geometry.TriangleMesh.create_cylinder(
            radius=plate_radius, height=plate_height
        )
        plate.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0]),
            center=[0, 0, 0]
        )
        plate.translate([0, plate_height / 2, 0])
        plate.paint_uniform_color(self.PLATE_COLOR)

        # Raised rim (thin cylinder around edge)
        rim_height = size * 0.03
        rim = o3d.geometry.TriangleMesh.create_cylinder(
            radius=plate_radius, height=rim_height
        )
        rim.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0]),
            center=[0, 0, 0]
        )
        rim.translate([0, plate_height + rim_height / 2, 0])
        rim.paint_uniform_color(self.PLATE_COLOR)

        combined = plate + rim
        return combined

    def create_bottle_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a tall vodka bottle shape."""
        size = self.spawn_point_size * 0.8

        # Bottle body (cylinder)
        body_radius = size * 0.15
        body_height = size * 0.5
        body = o3d.geometry.TriangleMesh.create_cylinder(
            radius=body_radius, height=body_height
        )
        body.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0]),
            center=[0, 0, 0]
        )
        body.translate([0, body_height / 2, 0])
        body.paint_uniform_color(self.BOTTLE_COLOR)

        # Bottle neck (narrower cylinder)
        neck_radius = body_radius * 0.4
        neck_height = size * 0.25
        neck = o3d.geometry.TriangleMesh.create_cylinder(
            radius=neck_radius, height=neck_height
        )
        neck.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0]),
            center=[0, 0, 0]
        )
        neck.translate([0, body_height + neck_height / 2, 0])
        neck.paint_uniform_color(self.BOTTLE_COLOR)

        # Shoulder (cone connecting body to neck)
        shoulder = o3d.geometry.TriangleMesh.create_cone(
            radius=body_radius, height=size * 0.1
        )
        shoulder.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0]),
            center=[0, 0, 0]
        )
        shoulder.translate([0, body_height, 0])
        shoulder.paint_uniform_color(self.BOTTLE_COLOR)

        combined = body + neck + shoulder
        return combined

    def create_ladder_mesh(self) -> o3d.geometry.TriangleMesh:
        """Create a ladder shape."""
        size = self.spawn_point_size * 1.2

        # Ladder dimensions
        rail_height = size * 1.0
        rail_width = size * 0.05
        rail_depth = size * 0.05
        rail_spacing = size * 0.3
        rung_count = 5

        combined = o3d.geometry.TriangleMesh()

        # Left rail
        left_rail = o3d.geometry.TriangleMesh.create_box(
            width=rail_width, height=rail_height, depth=rail_depth
        )
        left_rail.translate([-rail_spacing / 2 - rail_width / 2, 0, -rail_depth / 2])
        left_rail.paint_uniform_color(self.LADDER_COLOR)
        combined += left_rail

        # Right rail
        right_rail = o3d.geometry.TriangleMesh.create_box(
            width=rail_width, height=rail_height, depth=rail_depth
        )
        right_rail.translate([rail_spacing / 2 - rail_width / 2, 0, -rail_depth / 2])
        right_rail.paint_uniform_color(self.LADDER_COLOR)
        combined += right_rail

        # Rungs
        rung_width = rail_spacing
        rung_height = size * 0.04
        rung_depth = rail_depth
        for i in range(rung_count):
            rung = o3d.geometry.TriangleMesh.create_box(
                width=rung_width, height=rung_height, depth=rung_depth
            )
            y_pos = (i + 0.5) * rail_height / rung_count
            rung.translate([-rung_width / 2, y_pos, -rung_depth / 2])
            rung.paint_uniform_color(self.LADDER_COLOR)
            combined += rung

        return combined
