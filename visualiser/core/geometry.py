"""
Manages Open3D geometry objects for visualization
"""
from typing import Optional
from collections import defaultdict
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

    # Spawn type keys for template caching
    SPAWN_TYPE_DEFAULT = "default"
    SPAWN_TYPE_LEVEL_CHANGER = "level_changer"
    SPAWN_TYPE_LIGHTS = "lights"
    SPAWN_TYPE_PHYSIC = "physic"
    SPAWN_TYPE_SPACE_RESTRICTOR = "space_restrictor"
    SPAWN_TYPE_AMMO = "ammo"
    SPAWN_TYPE_INVENTORY_BOX = "inventory_box"
    SPAWN_TYPE_ZONE = "zone"
    SPAWN_TYPE_WEAPON = "weapon"
    SPAWN_TYPE_DETECTOR = "detector"
    SPAWN_TYPE_SMART_COVER = "smart_cover"
    SPAWN_TYPE_CAMPFIRE = "campfire"
    SPAWN_TYPE_BOX = "box"
    SPAWN_TYPE_MUG = "mug"
    SPAWN_TYPE_PLATE = "plate"
    SPAWN_TYPE_BOTTLE = "bottle"
    SPAWN_TYPE_LADDER = "ladder"

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
    LIGHTBULB_METAL_COLOR = [0.7, 0.7, 0.75]  # Silver/metal for bulb base
    BOX_WOOD_COLOR = [0.55, 0.35, 0.15]  # Medium brown wood
    MUG_COLOR = [0.85, 0.85, 0.8]  # Off-white ceramic
    PLATE_COLOR = [0.9, 0.9, 0.85]  # Light cream ceramic
    BOTTLE_COLOR = [0.3, 0.5, 0.3]  # Green glass
    LADDER_COLOR = [0.5, 0.35, 0.2]  # Wood brown

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
        self.level_vertex_highlight = None  # Separate point cloud for highlighted level vertices
        self._original_point_colors = None  # Store original colors for reset

        # Cached template meshes for highlights (created once, copied when needed)
        self._highlight_sphere_template = None
        self._spawn_highlight_template = None
        self._graph_highlight_template = None

        # Cached spawn templates by type (for batched mesh construction)
        self._spawn_templates = {}

        # Track currently highlighted level vertices for efficient color updates
        self._highlighted_level_vertices = set()

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
        self._create_level_vertex_highlight()

    def _get_spawn_type(self, section_name: str) -> str:
        """Get the spawn type key for a section name (used for template caching)."""
        if section_name == "level_changer":
            return self.SPAWN_TYPE_LEVEL_CHANGER
        elif section_name.startswith("lights_"):
            return self.SPAWN_TYPE_LIGHTS
        elif section_name.startswith("physic_") or section_name.startswith("explosive_"):
            return self.SPAWN_TYPE_PHYSIC
        elif section_name in ("space_restrictor", "smart_terrain"):
            return self.SPAWN_TYPE_SPACE_RESTRICTOR
        elif section_name.startswith("ammo_"):
            return self.SPAWN_TYPE_AMMO
        elif section_name.startswith("inventory_box"):
            return self.SPAWN_TYPE_INVENTORY_BOX
        elif section_name.startswith("zone_"):
            return self.SPAWN_TYPE_ZONE
        elif section_name.startswith("wpn_"):
            return self.SPAWN_TYPE_WEAPON
        elif section_name.startswith("detector_"):
            return self.SPAWN_TYPE_DETECTOR
        elif section_name == "smart_cover":
            return self.SPAWN_TYPE_SMART_COVER
        elif section_name == "campfire":
            return self.SPAWN_TYPE_CAMPFIRE
        elif section_name.startswith("box_"):
            return self.SPAWN_TYPE_BOX
        elif section_name == "krujka":
            return self.SPAWN_TYPE_MUG
        elif section_name in ("bludo", "kastrula", "vedro") or section_name.startswith("tarelka_"):
            return self.SPAWN_TYPE_PLATE
        elif section_name.startswith("bottle_") or section_name == "vodka":
            return self.SPAWN_TYPE_BOTTLE
        elif section_name == "table_lamp":
            return self.SPAWN_TYPE_LIGHTS
        elif section_name == "climable_object":
            return self.SPAWN_TYPE_LADDER
        else:
            return self.SPAWN_TYPE_DEFAULT

    def _get_spawn_template(self, spawn_type: str) -> o3d.geometry.TriangleMesh:
        """Get or create a cached spawn template mesh for the given type."""
        if spawn_type in self._spawn_templates:
            return self._spawn_templates[spawn_type]

        # Create template based on type
        if spawn_type == self.SPAWN_TYPE_LEVEL_CHANGER:
            template = self._create_level_changer_mesh()
        elif spawn_type == self.SPAWN_TYPE_LIGHTS:
            template = self._create_lightbulb_mesh()
        elif spawn_type == self.SPAWN_TYPE_PHYSIC:
            template = self._create_explosion_mesh()
        elif spawn_type == self.SPAWN_TYPE_SPACE_RESTRICTOR:
            template = self._create_space_restrictor_marker_mesh()
        elif spawn_type == self.SPAWN_TYPE_AMMO:
            template = self._create_ammo_mesh()
        elif spawn_type == self.SPAWN_TYPE_INVENTORY_BOX:
            template = self._create_inventory_box_mesh()
        elif spawn_type == self.SPAWN_TYPE_ZONE:
            template = self._create_zone_mesh()
        elif spawn_type == self.SPAWN_TYPE_WEAPON:
            template = self._create_weapon_mesh()
        elif spawn_type == self.SPAWN_TYPE_DETECTOR:
            template = self._create_detector_mesh()
        elif spawn_type == self.SPAWN_TYPE_SMART_COVER:
            template = self._create_smart_cover_mesh()
        elif spawn_type == self.SPAWN_TYPE_CAMPFIRE:
            template = self._create_campfire_mesh()
        elif spawn_type == self.SPAWN_TYPE_BOX:
            template = self._create_wooden_box_mesh()
        elif spawn_type == self.SPAWN_TYPE_MUG:
            template = self._create_mug_mesh()
        elif spawn_type == self.SPAWN_TYPE_PLATE:
            template = self._create_plate_mesh()
        elif spawn_type == self.SPAWN_TYPE_BOTTLE:
            template = self._create_bottle_mesh()
        elif spawn_type == self.SPAWN_TYPE_LADDER:
            template = self._create_ladder_mesh()
        else:
            template = self._create_default_spawn_mesh()

        self._spawn_templates[spawn_type] = template
        return template

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
        """Create lines for vertex connections using vectorized operations."""
        points = self.level_data.points
        vertex_count = len(self.level_data)

        # Bulk read all links in a single pass (much faster than per-vertex reads)
        all_links = self.level_data.get_all_links()  # (vertex_count, 4) array

        # Create source vertex indices repeated 4 times each (one per link slot)
        sources = np.repeat(np.arange(vertex_count, dtype=np.int32), 4)

        # Flatten links to match sources
        targets = all_links.flatten()

        # Filter to valid links only
        valid_mask = (targets != LevelData.INVALID_LINK) & (targets < vertex_count)
        valid_sources = sources[valid_mask]
        valid_targets = targets[valid_mask]

        # Stack into (N, 2) lines array
        if len(valid_sources) > 0:
            lines = np.stack([valid_sources, valid_targets], axis=1)
            # Pre-allocate colors using tile (faster than list comprehension)
            line_colors = np.tile([0.5, 0.5, 0.5], (len(lines), 1)).astype(np.float64)
        else:
            lines = np.zeros((0, 2), dtype=np.int32)
            line_colors = np.zeros((0, 3), dtype=np.float64)

        self.line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        self.line_set.colors = o3d.utility.Vector3dVector(line_colors)

    def _create_highlight_sphere(self):
        """Create the highlight sphere template and initial instance"""
        # Create template at origin (reused for all highlights)
        # Use resolution=12 for highlights (large, few)
        self._highlight_sphere_template = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.5, resolution=12
        )
        self._highlight_sphere_template.paint_uniform_color([1, 0, 0])
        self._highlight_sphere_template.compute_vertex_normals()

        # Create initial instance hidden off-screen
        self.highlight_sphere = o3d.geometry.TriangleMesh(self._highlight_sphere_template)
        self.highlight_sphere.translate([0, -10000, 0])

    def update_highlight_position(self, idx: int):
        """Update the highlight sphere position by copying cached template"""
        point = self.level_data.get_point(idx)
        if point is not None:
            # Copy template and translate to new position
            self.highlight_sphere = o3d.geometry.TriangleMesh(self._highlight_sphere_template)
            self.highlight_sphere.translate(point)
            return self.highlight_sphere
        return None

    def hide_highlight(self):
        """Hide the highlight sphere by returning a copy moved off-screen"""
        self.highlight_sphere = o3d.geometry.TriangleMesh(self._highlight_sphere_template)
        self.highlight_sphere.translate([0, -10000, 0])
        return self.highlight_sphere

    def _create_spawn_points(self):
        """Create meshes for spawn points using batched template construction.

        Groups spawns by type and batch constructs each group to avoid O(n²) mesh merging.
        """
        if self.spawn_data is None or len(self.spawn_data) == 0:
            self.spawn_points = o3d.geometry.TriangleMesh()
            return

        # Group spawns by type with their positions
        spawns_by_type = defaultdict(list)
        for i in range(len(self.spawn_data)):
            entity = self.spawn_data.get_entity(i)
            pos = self.spawn_data.get_position(i)
            if pos is None or entity is None:
                continue
            spawn_type = self._get_spawn_type(entity.section_name)
            spawns_by_type[spawn_type].append(pos)

        if not spawns_by_type:
            self.spawn_points = o3d.geometry.TriangleMesh()
            return

        # Pre-compute total vertices and triangles across all types
        all_verts_list = []
        all_tris_list = []
        all_colors_list = []
        vertex_offset = 0

        # Batch construct each type group
        for spawn_type, positions in spawns_by_type.items():
            if not positions:
                continue

            # Get cached template for this type
            template = self._get_spawn_template(spawn_type)
            template_verts = np.asarray(template.vertices)
            template_tris = np.asarray(template.triangles)

            # Get template colors (may have per-vertex colors)
            if template.has_vertex_colors():
                template_colors = np.asarray(template.vertex_colors)
            else:
                # Uniform color - this shouldn't happen but handle it
                template_colors = np.full((len(template_verts), 3), 0.5)

            n_verts = len(template_verts)
            n_tris = len(template_tris)
            n_spawns = len(positions)

            # Pre-allocate arrays for this type group
            group_verts = np.empty((n_spawns * n_verts, 3), dtype=np.float64)
            group_tris = np.empty((n_spawns * n_tris, 3), dtype=np.int32)
            group_colors = np.empty((n_spawns * n_verts, 3), dtype=np.float64)

            for i, pos in enumerate(positions):
                v_start = i * n_verts
                t_start = i * n_tris

                # Translate template vertices to position
                group_verts[v_start:v_start + n_verts] = template_verts + pos

                # Offset triangle indices (relative to this group)
                group_tris[t_start:t_start + n_tris] = template_tris + v_start

                # Copy template colors
                group_colors[v_start:v_start + n_verts] = template_colors

            # Offset triangles by global vertex offset and append to lists
            group_tris += vertex_offset
            all_verts_list.append(group_verts)
            all_tris_list.append(group_tris)
            all_colors_list.append(group_colors)
            vertex_offset += len(group_verts)

        # Combine all groups into single mesh
        if all_verts_list:
            all_verts = np.vstack(all_verts_list)
            all_tris = np.vstack(all_tris_list)
            all_colors = np.vstack(all_colors_list)

            self.spawn_points = o3d.geometry.TriangleMesh()
            self.spawn_points.vertices = o3d.utility.Vector3dVector(all_verts)
            self.spawn_points.triangles = o3d.utility.Vector3iVector(all_tris)
            self.spawn_points.vertex_colors = o3d.utility.Vector3dVector(all_colors)
            self.spawn_points.compute_vertex_normals()
        else:
            self.spawn_points = o3d.geometry.TriangleMesh()

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
        """Create lightbulb shape with yellow glass bulb and metal base"""
        size = self.SPAWN_POINT_SIZE

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

    def _create_wooden_box_mesh(self):
        """Create a small wooden box/crate shape"""
        size = self.SPAWN_POINT_SIZE * 0.7

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

    def _create_mug_mesh(self):
        """Create a small mug/cup shape"""
        size = self.SPAWN_POINT_SIZE * 0.5

        # Mug body (cylinder) - created along Z, rotate to Y-up
        body_radius = size * 0.3
        body_height = size * 0.5
        body = o3d.geometry.TriangleMesh.create_cylinder(
            radius=body_radius, height=body_height
        )
        # Rotate from Z-axis to Y-axis (upright)
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

    def _create_plate_mesh(self):
        """Create a small plate/dish shape"""
        size = self.SPAWN_POINT_SIZE * 0.6

        # Plate base (flat cylinder) - rotate to be horizontal
        plate_radius = size * 0.4
        plate_height = size * 0.05
        plate = o3d.geometry.TriangleMesh.create_cylinder(
            radius=plate_radius, height=plate_height
        )
        # Rotate from Z-axis to Y-axis (flat on ground)
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

    def _create_bottle_mesh(self):
        """Create a tall vodka bottle shape"""
        size = self.SPAWN_POINT_SIZE * 0.8

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

    def _create_ladder_mesh(self):
        """Create a ladder shape"""
        size = self.SPAWN_POINT_SIZE * 1.2

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
        """Create the highlight cube template and initial instance"""
        # Create template centered at origin (reused for all highlights)
        self._spawn_highlight_template = o3d.geometry.TriangleMesh.create_box(
            width=self.SPAWN_HIGHLIGHT_SIZE,
            height=self.SPAWN_HIGHLIGHT_SIZE,
            depth=self.SPAWN_HIGHLIGHT_SIZE
        )
        # Center the template on origin
        self._spawn_highlight_template.translate(np.array([
            -self.SPAWN_HIGHLIGHT_SIZE / 2,
            -self.SPAWN_HIGHLIGHT_SIZE / 2,
            -self.SPAWN_HIGHLIGHT_SIZE / 2
        ]))
        self._spawn_highlight_template.paint_uniform_color(self.SPAWN_HIGHLIGHT_COLOR)
        self._spawn_highlight_template.compute_vertex_normals()

        # Create initial instance hidden off-screen
        self.spawn_highlight = o3d.geometry.TriangleMesh(self._spawn_highlight_template)
        self.spawn_highlight.translate([0, -10000, 0])

    def update_spawn_highlight_position(self, idx: int):
        """Update the spawn highlight cube position by copying cached template"""
        if self.spawn_data is None:
            return None

        pos = self.spawn_data.get_position(idx)
        if pos is not None:
            # Copy template (already centered) and translate to position
            self.spawn_highlight = o3d.geometry.TriangleMesh(self._spawn_highlight_template)
            self.spawn_highlight.translate(pos)
            return self.spawn_highlight
        return None

    def hide_spawn_highlight(self):
        """Hide the spawn highlight by returning a copy moved off-screen"""
        self.spawn_highlight = o3d.geometry.TriangleMesh(self._spawn_highlight_template)
        self.spawn_highlight.translate([0, -10000, 0])
        return self.spawn_highlight

    def _create_graph_vertices(self):
        """Create spheres for game graph vertices using batched template construction.

        Uses a single template sphere and pre-allocated arrays to avoid O(n²) mesh merging.
        """
        if self.graph_data is None or len(self.graph_data) == 0:
            self.graph_vertices = o3d.geometry.TriangleMesh()
            return

        # Get positions array and filter out None positions
        positions = self.graph_data.positions
        inter_level_flags = self.graph_data.inter_level_flags
        n = len(positions)

        if n == 0:
            self.graph_vertices = o3d.geometry.TriangleMesh()
            return

        # Create low-resolution template sphere at origin (resolution=8 for small spheres)
        template = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.GRAPH_VERTEX_SIZE, resolution=8
        )
        template_verts = np.asarray(template.vertices)
        template_tris = np.asarray(template.triangles)
        n_verts = len(template_verts)
        n_tris = len(template_tris)

        # Pre-allocate arrays for all spheres
        all_verts = np.empty((n * n_verts, 3), dtype=np.float64)
        all_tris = np.empty((n * n_tris, 3), dtype=np.int32)
        all_colors = np.empty((n * n_verts, 3), dtype=np.float64)

        # Batch construct all spheres
        for i in range(n):
            v_start = i * n_verts
            t_start = i * n_tris

            # Translate template vertices to position
            all_verts[v_start:v_start + n_verts] = template_verts + positions[i]

            # Offset triangle indices
            all_tris[t_start:t_start + n_tris] = template_tris + v_start

            # Set color based on inter-level status
            if i < len(inter_level_flags) and inter_level_flags[i]:
                color = self.GRAPH_VERTEX_INTER_LEVEL_COLOR
            else:
                color = self.GRAPH_VERTEX_COLOR
            all_colors[v_start:v_start + n_verts] = color

        # Create single mesh from arrays
        self.graph_vertices = o3d.geometry.TriangleMesh()
        self.graph_vertices.vertices = o3d.utility.Vector3dVector(all_verts)
        self.graph_vertices.triangles = o3d.utility.Vector3iVector(all_tris)
        self.graph_vertices.vertex_colors = o3d.utility.Vector3dVector(all_colors)
        self.graph_vertices.compute_vertex_normals()

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
        # Use np.tile for efficient color array creation
        line_colors = np.tile(self.GRAPH_EDGE_COLOR, (len(edges), 1)).astype(np.float64)

        self.graph_edges = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines),
        )
        self.graph_edges.colors = o3d.utility.Vector3dVector(line_colors)

    def _create_graph_highlight(self):
        """Create the highlight sphere template and initial instance"""
        # Create template at origin (reused for all highlights)
        # Use resolution=12 for highlights (large, few)
        self._graph_highlight_template = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.GRAPH_HIGHLIGHT_SIZE, resolution=12
        )
        self._graph_highlight_template.paint_uniform_color(self.GRAPH_HIGHLIGHT_COLOR)
        self._graph_highlight_template.compute_vertex_normals()

        # Create initial instance hidden off-screen
        self.graph_highlight = o3d.geometry.TriangleMesh(self._graph_highlight_template)
        self.graph_highlight.translate([0, -10000, 0])

    def update_graph_highlight_position(self, idx: int):
        """Update the graph highlight sphere position by copying cached template"""
        if self.graph_data is None:
            return None

        pos = self.graph_data.get_position(idx)
        if pos is not None:
            # Copy template and translate to new position
            self.graph_highlight = o3d.geometry.TriangleMesh(self._graph_highlight_template)
            self.graph_highlight.translate(pos)
            return self.graph_highlight
        return None

    def hide_graph_highlight(self):
        """Hide the graph highlight by returning a copy moved off-screen"""
        self.graph_highlight = o3d.geometry.TriangleMesh(self._graph_highlight_template)
        self.graph_highlight.translate([0, -10000, 0])
        return self.graph_highlight

    def _create_level_vertex_highlight(self):
        """Create an empty point cloud for level vertex highlighting."""
        self.level_vertex_highlight = o3d.geometry.PointCloud()

    def highlight_level_vertices(self, vertex_ids: set):
        """Highlight specific level vertices by creating a separate overlay point cloud.

        This is much faster than modifying the main point cloud colors.

        Args:
            vertex_ids: Set of level vertex IDs to highlight in red
        """
        if not vertex_ids:
            # Clear highlight - create empty point cloud
            self.level_vertex_highlight = o3d.geometry.PointCloud()
            self._highlighted_level_vertices = set()
            return

        # Get positions for highlighted vertices
        highlighted_points = []
        for vid in vertex_ids:
            if 0 <= vid < len(self.level_data.points):
                highlighted_points.append(self.level_data.points[vid])

        if not highlighted_points:
            self.level_vertex_highlight = o3d.geometry.PointCloud()
            self._highlighted_level_vertices = set()
            return

        # Create new point cloud with just the highlighted vertices
        points_array = np.array(highlighted_points)
        # Use np.tile for efficient color array creation
        colors_array = np.tile(self.LINKED_HIGHLIGHT_COLOR, (len(highlighted_points), 1)).astype(np.float64)

        self.level_vertex_highlight = o3d.geometry.PointCloud()
        self.level_vertex_highlight.points = o3d.utility.Vector3dVector(points_array)
        self.level_vertex_highlight.colors = o3d.utility.Vector3dVector(colors_array)

        # Track highlighted vertices
        self._highlighted_level_vertices = vertex_ids.copy()

    def reset_level_vertex_colors(self):
        """Clear the level vertex highlight overlay."""
        self.level_vertex_highlight = o3d.geometry.PointCloud()
        self._highlighted_level_vertices = set()
