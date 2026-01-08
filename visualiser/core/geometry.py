"""
Manages Open3D geometry objects for visualization
"""
from typing import Optional
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

from core.data_loader import LevelData, SpawnData, GraphData

# Import shape parsing from shared parsers
import sys
from pathlib import Path
compiler_path = str(Path(__file__).parent.parent.parent / "compiler")
if compiler_path not in sys.path:
    sys.path.append(compiler_path)
from parsers import parse_entity_shapes


class GeometryManager:
    """Creates and manages Open3D geometry objects"""

    # Spawn point rendering constants
    SPAWN_COLOR = [0, 0.8, 0]  # Green
    SPAWN_HIGHLIGHT_COLOR = [1, 1, 0]  # Yellow
    SPAWN_POINT_SIZE = 0.5
    SPAWN_HIGHLIGHT_SIZE = 0.7

    # Special spawn type colors
    LEVEL_CHANGER_COLOR = [1.0, 0.5, 0.0]  # Orange
    LIGHTS_COLOR = [1.0, 1.0, 0.0]  # Yellow
    PHYSIC_COLOR = [1.0, 0.3, 0.0]  # Red-orange
    SPACE_RESTRICTOR_COLOR = [0.2, 0.8, 0.2]  # Light green
    SPACE_RESTRICTOR_SHAPE_COLOR = [0.0, 1.0, 0.5]  # Cyan-green for shapes
    AMMO_COLOR = [0.8, 0.6, 0.2]  # Brass/bronze
    INVENTORY_BOX_COLOR = [1.0, 0.84, 0.0]  # Gold
    ZONE_COLOR = [0.2, 0.6, 1.0]  # Bright blue
    WEAPON_COLOR = [0.3, 0.3, 0.35]  # Dark gunmetal grey
    DETECTOR_COLOR = [0.2, 0.2, 0.2]  # Dark grey body
    DETECTOR_SCREEN_COLOR = [0.1, 0.4, 0.1]  # Dark green screen
    DETECTOR_BUTTON_COLORS = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.8, 0.8, 0.2]]  # Red, green, yellow
    SMART_COVER_COLOR = [0.6, 0.5, 0.35]  # Sandy brown for sandbags
    CAMPFIRE_WOOD_COLOR = [0.4, 0.25, 0.1]  # Dark brown for logs
    CAMPFIRE_FLAME_COLOR = [1.0, 0.5, 0.0]  # Orange flames
    CAMPFIRE_EMBER_COLOR = [1.0, 0.2, 0.0]  # Red-orange embers

    # Game graph rendering constants
    GRAPH_VERTEX_COLOR = [0.2, 0.5, 1.0]  # Bright blue
    GRAPH_VERTEX_INTER_LEVEL_COLOR = [0.5, 0.0, 0.8]  # Deep purple
    GRAPH_EDGE_COLOR = [0.3, 0.6, 1.0]  # Blue
    GRAPH_HIGHLIGHT_COLOR = [1.0, 0.5, 0.0]  # Orange
    GRAPH_VERTEX_SIZE = 1.0
    GRAPH_HIGHLIGHT_SIZE = 1.3

    # Linked graph vertex highlighting
    LINKED_HIGHLIGHT_COLOR = [1.0, 0.0, 0.0]  # Bright red

    def __init__(self, level_data: LevelData, spawn_data: Optional[SpawnData] = None,
                 graph_data: Optional[GraphData] = None):
        self.level_data = level_data
        self.spawn_data = spawn_data
        self.graph_data = graph_data
        self.point_cloud = None
        self.line_set = None
        self.highlight_sphere = None
        self.spawn_points = None
        self.spawn_highlight = None
        self.space_restrictor_shapes = None  # Semi-transparent shape meshes
        self.graph_vertices = None  # Game graph vertex spheres
        self.graph_edges = None  # Game graph edge lines
        self.graph_highlight = None  # Game graph highlight sphere
        self._original_point_colors = None  # Store original colors for reset

        self._create_geometries()

    def _create_geometries(self):
        """Create all initial geometries"""
        self._create_point_cloud()
        self._create_line_set()
        self._create_highlight_sphere()
        self._create_spawn_points()
        self._create_spawn_highlight()
        self._create_space_restrictor_shapes()
        self._create_graph_vertices()
        self._create_graph_edges()
        self._create_graph_highlight()

    def _create_point_cloud(self):
        """Create the point cloud from vertex data"""
        points = self.level_data.points
        colors = self.level_data.colors

        # Normalize colors to [0,1] for Open3D
        color_min = colors.min()
        color_max = colors.max()
        if color_max > color_min:
            colors_normalized = (colors - color_min) / (color_max - color_min)
        else:
            colors_normalized = np.zeros_like(colors)

        # Convert to RGB using a colormap (plasma)
        colors_rgb = cm.plasma(colors_normalized)[:, :3]

        # Store original colors for reset
        self._original_point_colors = colors_rgb.copy()

        # Create Open3D point cloud
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors_rgb)

    def _create_line_set(self):
        """Create lines for vertex connections"""
        points = self.level_data.points
        vertex_count = len(self.level_data)

        lines = []
        line_colors = []

        for i in range(vertex_count):
            links = self.level_data.get_links(i)
            for link_id in links:
                if link_id != LevelData.INVALID_LINK and link_id < vertex_count:
                    lines.append([i, link_id])
                    line_colors.append([0.5, 0.5, 0.5])  # grey lines

        lines = np.array(lines) if lines else np.zeros((0, 2), dtype=np.int32)
        line_colors = np.array(line_colors) if line_colors else np.zeros((0, 3))

        self.line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        self.line_set.colors = o3d.utility.Vector3dVector(line_colors)

    def _create_highlight_sphere(self):
        """Create the highlight sphere for selected vertices"""
        self.highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        self.highlight_sphere.paint_uniform_color([1, 0, 0])
        self.highlight_sphere.translate([0, -10000, 0])  # Hide initially

    def update_highlight_position(self, idx: int):
        """Update the highlight sphere position"""
        point = self.level_data.get_point(idx)
        if point is not None:
            # Create new sphere at position
            self.highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            self.highlight_sphere.paint_uniform_color([1, 0, 0])
            self.highlight_sphere.translate(point)
            return self.highlight_sphere
        return None

    def _create_spawn_points(self):
        """Create meshes for spawn points with different shapes based on type"""
        if self.spawn_data is None or len(self.spawn_data) == 0:
            # Create empty mesh to avoid issues
            self.spawn_points = o3d.geometry.TriangleMesh()
            return

        # Combine all spawn point meshes into one
        combined_mesh = o3d.geometry.TriangleMesh()

        for i in range(len(self.spawn_data)):
            entity = self.spawn_data.get_entity(i)
            pos = self.spawn_data.get_position(i)
            if pos is None or entity is None:
                continue

            section_name = entity.section_name

            if section_name == "level_changer":
                # Large orange sphere (1/3 larger than default)
                mesh = self._create_level_changer_mesh()
            elif section_name.startswith("lights_"):
                # Yellow lightbulb
                mesh = self._create_lightbulb_mesh()
            elif section_name.startswith("physic_") or section_name.startswith("explosive_"):
                # Spikey explosion
                mesh = self._create_explosion_mesh()
            elif section_name in ("space_restrictor", "smart_terrain"):
                # Small green sphere (shapes rendered separately with transparency)
                mesh = self._create_space_restrictor_marker_mesh()
            elif section_name.startswith("ammo_"):
                # Brass bullet pile
                mesh = self._create_ammo_mesh()
            elif section_name.startswith("inventory_box"):
                # Gold chest/crate
                mesh = self._create_inventory_box_mesh()
            elif section_name.startswith("zone_"):
                # Bright blue lightning bolt
                mesh = self._create_zone_mesh()
            elif section_name.startswith("wpn_"):
                # Dark gunmetal pistol
                mesh = self._create_weapon_mesh()
            elif section_name.startswith("detector_"):
                # Detector device (TV remote style)
                mesh = self._create_detector_mesh()
            elif section_name == "smart_cover":
                # Sandbag barricade
                mesh = self._create_smart_cover_mesh()
            elif section_name == "campfire":
                # Campfire with logs and flames
                mesh = self._create_campfire_mesh()
            else:
                # Default green box
                mesh = self._create_default_spawn_mesh()

            # Center and position the mesh
            mesh.translate(pos)
            combined_mesh += mesh

        self.spawn_points = combined_mesh

    def _create_default_spawn_mesh(self):
        """Create default green box for regular spawns"""
        box = o3d.geometry.TriangleMesh.create_box(
            width=self.SPAWN_POINT_SIZE,
            height=self.SPAWN_POINT_SIZE,
            depth=self.SPAWN_POINT_SIZE
        )
        # Center the box on origin
        box.translate(np.array([
            -self.SPAWN_POINT_SIZE / 2,
            -self.SPAWN_POINT_SIZE / 2,
            -self.SPAWN_POINT_SIZE / 2
        ]))
        box.paint_uniform_color(self.SPAWN_COLOR)
        return box

    def _create_level_changer_mesh(self):
        """Create large orange sphere for level changers"""
        # 1.5x larger than default size
        radius = self.SPAWN_POINT_SIZE * 1.5 / 2
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color(self.LEVEL_CHANGER_COLOR)
        return sphere

    def _create_lightbulb_mesh(self):
        """Create yellow lightbulb shape for lights"""
        size = self.SPAWN_POINT_SIZE

        # Bulb part (sphere) - the glass part
        bulb_radius = size * 0.35
        bulb = o3d.geometry.TriangleMesh.create_sphere(radius=bulb_radius)
        bulb.translate([0, size * 0.15, 0])

        # Screw base (cylinder narrowing down)
        base_radius = bulb_radius * 0.6
        base_height = size * 0.25
        base = o3d.geometry.TriangleMesh.create_cylinder(radius=base_radius, height=base_height)
        base.translate([0, -size * 0.1, 0])

        # Bottom tip
        tip_radius = base_radius * 0.4
        tip_height = size * 0.15
        tip = o3d.geometry.TriangleMesh.create_cone(radius=tip_radius, height=tip_height)
        # Flip the cone so it points down
        tip.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0]), center=[0, 0, 0])
        tip.translate([0, -size * 0.3, 0])

        # Combine
        lightbulb = bulb + base + tip
        lightbulb.paint_uniform_color(self.LIGHTS_COLOR)
        return lightbulb

    def _create_explosion_mesh(self):
        """Create spikey explosion shape for physics objects"""
        # Create an icosahedron and scale its vertices outward for spiky effect
        # 40% bigger than original
        size = self.SPAWN_POINT_SIZE * 0.6 * 1.4

        # Start with icosahedron
        mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius=size * 0.5)

        # Get vertices and make them spikier by scaling outward
        vertices = np.asarray(mesh.vertices)

        # Add spikes by duplicating vertices and pushing them outward
        spike_mesh = o3d.geometry.TriangleMesh()

        # Create spikes from center to each vertex
        center = np.array([0, 0, 0])
        spike_length = size * 0.8

        for vertex in vertices:
            direction = vertex / np.linalg.norm(vertex)
            spike_tip = direction * spike_length

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

    def _create_space_restrictor_marker_mesh(self):
        """Create small green sphere marker for space restrictor spawns"""
        radius = self.SPAWN_POINT_SIZE * 0.3
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color(self.SPACE_RESTRICTOR_COLOR)
        return sphere

    def _create_ammo_mesh(self):
        """Create a small pile of bullet/shell-like cylinders in brass color"""
        size = self.SPAWN_POINT_SIZE * 0.6
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

    def _create_inventory_box_mesh(self):
        """Create a rectangular chest/crate shape in gold"""
        size = self.SPAWN_POINT_SIZE

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

    def _create_zone_mesh(self):
        """Create a high-fidelity lightning bolt shape in bright blue"""
        size = self.SPAWN_POINT_SIZE * 1.5

        # Define lightning bolt vertices as a 2D polygon, then extrude
        # Classic zigzag lightning bolt shape
        thickness = size * 0.08  # Depth of the bolt

        # Lightning bolt profile points (x, y) - a proper jagged bolt shape
        # Starting from top, going clockwise
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

        # Triangulate the front and back faces (fan triangulation from center)
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

    def _create_weapon_mesh(self):
        """Create a small pistol shape in gunmetal grey"""
        size = self.SPAWN_POINT_SIZE * 0.8
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

    def _create_detector_mesh(self):
        """Create a detector device (TV remote style) with colored buttons"""
        size = self.SPAWN_POINT_SIZE * 0.7
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

    def _create_smart_cover_mesh(self):
        """Create a sandbag barricade shape"""
        size = self.SPAWN_POINT_SIZE * 1.2
        combined = o3d.geometry.TriangleMesh()

        # Individual sandbag dimensions
        bag_width = size * 0.35
        bag_height = size * 0.18
        bag_depth = size * 0.18

        def create_sandbag(width, height, depth):
            """Create a single sandbag (rounded box shape using a squashed sphere)"""
            # Use an ellipsoid for a more organic sandbag look
            bag = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=8)
            # Scale to sandbag proportions
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

    def _create_campfire_mesh(self):
        """Create a campfire with logs arranged in a circle and flames"""
        size = self.SPAWN_POINT_SIZE * 1.7  # 70% bigger
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
        # Main central flame - rotate from Z-axis to Y-axis
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

    def _create_space_restrictor_shapes(self):
        """Initialize empty space_restrictor_shapes - shapes shown on selection only"""
        self.space_restrictor_shapes = o3d.geometry.TriangleMesh()

    def get_shapes_for_spawn(self, idx: int):
        """Create shape meshes for a specific spawn entity (used when selected)"""
        if self.spawn_data is None:
            return None

        entity = self.spawn_data.get_entity(idx)
        pos = self.spawn_data.get_position(idx)
        if pos is None or entity is None:
            return None

        if entity.section_name not in ("space_restrictor", "smart_terrain"):
            return None

        # Parse shapes from the entity
        shapes = parse_entity_shapes(entity)
        if not shapes:
            return None

        combined_mesh = o3d.geometry.TriangleMesh()

        for shape in shapes:
            if shape.is_sphere and shape.center is not None and shape.radius is not None:
                # Create sphere at entity position + shape center (with Z mirrored)
                shape_center = np.array([
                    shape.center[0],
                    shape.center[1],
                    -shape.center[2]  # Mirror Z to match display coordinates
                ])
                world_pos = pos + shape_center

                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=shape.radius)
                sphere.paint_uniform_color(self.SPACE_RESTRICTOR_SHAPE_COLOR)
                sphere.translate(world_pos)
                combined_mesh += sphere

            elif shape.is_box and shape.box_axes is not None and shape.box_translation is not None:
                # The box_axes are 3 axis vectors scaled by half-extents
                box = o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=2)
                box.translate(np.array([-1, -1, -1]))  # Center on origin
                box.paint_uniform_color(self.SPACE_RESTRICTOR_SHAPE_COLOR)

                # Build transformation matrix from axes
                axes = shape.box_axes
                trans = shape.box_translation

                transform = np.eye(4)
                transform[0, 0:3] = [axes[0][0], axes[1][0], axes[2][0]]
                transform[1, 0:3] = [axes[0][1], axes[1][1], axes[2][1]]
                transform[2, 0:3] = [axes[0][2], axes[1][2], axes[2][2]]
                transform[0, 3] = trans[0]
                transform[1, 3] = trans[1]
                transform[2, 3] = -trans[2]

                box.transform(transform)
                box.translate(pos)
                combined_mesh += box

        return combined_mesh if len(combined_mesh.vertices) > 0 else None

    def hide_space_restrictor_shapes(self):
        """Return an empty mesh to hide the shapes"""
        self.space_restrictor_shapes = o3d.geometry.TriangleMesh()
        return self.space_restrictor_shapes

    def _create_spawn_highlight(self):
        """Create the highlight cube for selected spawn objects"""
        self.spawn_highlight = o3d.geometry.TriangleMesh.create_box(
            width=self.SPAWN_HIGHLIGHT_SIZE,
            height=self.SPAWN_HIGHLIGHT_SIZE,
            depth=self.SPAWN_HIGHLIGHT_SIZE
        )
        self.spawn_highlight.paint_uniform_color(self.SPAWN_HIGHLIGHT_COLOR)
        # Hide initially (move far off-screen)
        self.spawn_highlight.translate([0, -10000, 0])

    def update_spawn_highlight_position(self, idx: int):
        """Update the spawn highlight cube position"""
        if self.spawn_data is None:
            return None

        pos = self.spawn_data.get_position(idx)
        if pos is not None:
            # Create new box at position
            self.spawn_highlight = o3d.geometry.TriangleMesh.create_box(
                width=self.SPAWN_HIGHLIGHT_SIZE,
                height=self.SPAWN_HIGHLIGHT_SIZE,
                depth=self.SPAWN_HIGHLIGHT_SIZE
            )
            self.spawn_highlight.paint_uniform_color(self.SPAWN_HIGHLIGHT_COLOR)
            # Center the box on the position
            self.spawn_highlight.translate(pos - np.array([
                self.SPAWN_HIGHLIGHT_SIZE / 2,
                self.SPAWN_HIGHLIGHT_SIZE / 2,
                self.SPAWN_HIGHLIGHT_SIZE / 2
            ]))
            return self.spawn_highlight
        return None

    def hide_spawn_highlight(self):
        """Hide the spawn highlight by moving it off-screen"""
        self.spawn_highlight = o3d.geometry.TriangleMesh.create_box(
            width=self.SPAWN_HIGHLIGHT_SIZE,
            height=self.SPAWN_HIGHLIGHT_SIZE,
            depth=self.SPAWN_HIGHLIGHT_SIZE
        )
        self.spawn_highlight.paint_uniform_color(self.SPAWN_HIGHLIGHT_COLOR)
        self.spawn_highlight.translate([0, -10000, 0])
        return self.spawn_highlight

    def _create_graph_vertices(self):
        """Create spheres for game graph vertices with colors based on inter-level status"""
        if self.graph_data is None or len(self.graph_data) == 0:
            self.graph_vertices = o3d.geometry.TriangleMesh()
            return

        combined_mesh = o3d.geometry.TriangleMesh()
        inter_level_flags = self.graph_data.inter_level_flags

        for i in range(len(self.graph_data)):
            pos = self.graph_data.get_position(i)
            if pos is None:
                continue

            # Create sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.GRAPH_VERTEX_SIZE)

            # Color based on inter-level status
            if i < len(inter_level_flags) and inter_level_flags[i]:
                sphere.paint_uniform_color(self.GRAPH_VERTEX_INTER_LEVEL_COLOR)
            else:
                sphere.paint_uniform_color(self.GRAPH_VERTEX_COLOR)

            sphere.translate(pos)
            combined_mesh += sphere

        self.graph_vertices = combined_mesh

    def _create_graph_edges(self):
        """Create line set for intra-level game graph edges"""
        if self.graph_data is None or len(self.graph_data) == 0:
            self.graph_edges = o3d.geometry.LineSet()
            return

        positions = self.graph_data.positions
        edges = self.graph_data.intra_level_edges

        if len(edges) == 0:
            self.graph_edges = o3d.geometry.LineSet()
            return

        lines = np.array(edges, dtype=np.int32)
        line_colors = np.array([self.GRAPH_EDGE_COLOR for _ in range(len(edges))])

        self.graph_edges = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines),
        )
        self.graph_edges.colors = o3d.utility.Vector3dVector(line_colors)

    def _create_graph_highlight(self):
        """Create the highlight sphere for selected graph vertices"""
        self.graph_highlight = o3d.geometry.TriangleMesh.create_sphere(radius=self.GRAPH_HIGHLIGHT_SIZE)
        self.graph_highlight.paint_uniform_color(self.GRAPH_HIGHLIGHT_COLOR)
        self.graph_highlight.translate([0, -10000, 0])  # Hide initially

    def update_graph_highlight_position(self, idx: int):
        """Update the graph highlight sphere position"""
        if self.graph_data is None:
            return None

        pos = self.graph_data.get_position(idx)
        if pos is not None:
            self.graph_highlight = o3d.geometry.TriangleMesh.create_sphere(radius=self.GRAPH_HIGHLIGHT_SIZE)
            self.graph_highlight.paint_uniform_color(self.GRAPH_HIGHLIGHT_COLOR)
            self.graph_highlight.translate(pos)
            return self.graph_highlight
        return None

    def hide_graph_highlight(self):
        """Hide the graph highlight by moving it off-screen"""
        self.graph_highlight = o3d.geometry.TriangleMesh.create_sphere(radius=self.GRAPH_HIGHLIGHT_SIZE)
        self.graph_highlight.paint_uniform_color(self.GRAPH_HIGHLIGHT_COLOR)
        self.graph_highlight.translate([0, -10000, 0])
        return self.graph_highlight

    def highlight_level_vertices(self, vertex_ids: set):
        """Highlight specific level vertices in red, keep others normal.

        Args:
            vertex_ids: Set of level vertex IDs to highlight in red
        """
        if self._original_point_colors is None:
            return

        # Start with original colors
        new_colors = self._original_point_colors.copy()

        # Set highlighted vertices to red
        for vid in vertex_ids:
            if 0 <= vid < len(new_colors):
                new_colors[vid] = self.LINKED_HIGHLIGHT_COLOR

        # Update point cloud colors
        self.point_cloud.colors = o3d.utility.Vector3dVector(new_colors)

    def reset_level_vertex_colors(self):
        """Reset all level vertex colors to normal cover score gradient."""
        if self._original_point_colors is None:
            return

        # Restore original plasma colormap colors
        self.point_cloud.colors = o3d.utility.Vector3dVector(self._original_point_colors.copy())
