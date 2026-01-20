"""
Manages Open3D geometry objects for visualization
"""
from typing import Optional
from collections import defaultdict
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

from core.data_loader import LevelData, SpawnData, GraphData, PatrolData
from core.spawn_meshes import SpawnMeshFactory

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
    SPAWN_POINT_SIZE = 0.5
    SPAWN_HIGHLIGHT_COLOR = [1, 1, 0]  # Yellow
    SPAWN_HIGHLIGHT_SIZE = 0.7

    # Space restrictor shape color (used for selection display)
    SPACE_RESTRICTOR_SHAPE_COLOR = [0.0, 1.0, 0.5]  # Cyan-green for shapes

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

    # Game graph rendering constants
    GRAPH_VERTEX_COLOR = [0.2, 0.5, 1.0]  # Bright blue
    GRAPH_VERTEX_INTER_LEVEL_COLOR = [0.5, 0.0, 0.8]  # Deep purple
    GRAPH_EDGE_COLOR = [0.3, 0.6, 1.0]  # Blue
    GRAPH_HIGHLIGHT_COLOR = [1.0, 0.5, 0.0]  # Orange
    GRAPH_VERTEX_SIZE = 1.0
    GRAPH_HIGHLIGHT_SIZE = 1.3

    # Linked graph vertex highlighting
    LINKED_HIGHLIGHT_COLOR = [1.0, 0.0, 0.0]  # Bright red

    # Patrol path rendering constants
    PATROL_POINT_COLOR = [0.1, 0.1, 0.1]  # Near-black
    PATROL_EDGE_COLOR = [0.1, 0.1, 0.1]   # Near-black
    PATROL_HIGHLIGHT_COLOR = [1.0, 0.5, 0.0]  # Orange
    PATROL_POINT_SIZE = 0.2  # Small spheres
    PATROL_HIGHLIGHT_SIZE = 0.3

    def __init__(self, level_data: LevelData, spawn_data: Optional[SpawnData] = None,
                 graph_data: Optional[GraphData] = None, patrol_data: Optional[PatrolData] = None):
        self.level_data = level_data
        self.spawn_data = spawn_data
        self.graph_data = graph_data
        self.patrol_data = patrol_data
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
        self.patrol_points = None  # Patrol path node spheres
        self.patrol_edges = None  # Patrol path connection lines
        self.patrol_highlight = None  # Patrol point highlight sphere
        self._original_point_colors = None  # Store original colors for reset

        # Cached template meshes for highlights (created once, copied when needed)
        self._highlight_sphere_template = None
        self._spawn_highlight_template = None
        self._graph_highlight_template = None
        self._patrol_highlight_template = None

        # Spawn mesh factory for creating spawn point meshes
        self._spawn_mesh_factory = SpawnMeshFactory(self.SPAWN_POINT_SIZE)

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
        self._create_patrol_points()
        self._create_patrol_edges()
        self._create_patrol_highlight()

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

        # Map spawn types to factory methods
        factory = self._spawn_mesh_factory
        factory_methods = {
            self.SPAWN_TYPE_LEVEL_CHANGER: factory.create_level_changer_mesh,
            self.SPAWN_TYPE_LIGHTS: factory.create_lightbulb_mesh,
            self.SPAWN_TYPE_PHYSIC: factory.create_explosion_mesh,
            self.SPAWN_TYPE_SPACE_RESTRICTOR: factory.create_space_restrictor_marker_mesh,
            self.SPAWN_TYPE_AMMO: factory.create_ammo_mesh,
            self.SPAWN_TYPE_INVENTORY_BOX: factory.create_inventory_box_mesh,
            self.SPAWN_TYPE_ZONE: factory.create_zone_mesh,
            self.SPAWN_TYPE_WEAPON: factory.create_weapon_mesh,
            self.SPAWN_TYPE_DETECTOR: factory.create_detector_mesh,
            self.SPAWN_TYPE_SMART_COVER: factory.create_smart_cover_mesh,
            self.SPAWN_TYPE_CAMPFIRE: factory.create_campfire_mesh,
            self.SPAWN_TYPE_BOX: factory.create_wooden_box_mesh,
            self.SPAWN_TYPE_MUG: factory.create_mug_mesh,
            self.SPAWN_TYPE_PLATE: factory.create_plate_mesh,
            self.SPAWN_TYPE_BOTTLE: factory.create_bottle_mesh,
            self.SPAWN_TYPE_LADDER: factory.create_ladder_mesh,
        }

        # Get factory method or use default
        create_method = factory_methods.get(spawn_type, factory.create_default_spawn_mesh)
        template = create_method()

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

    def _create_highlight_template(self, radius, color, resolution=12):
        """Create a highlight sphere template at origin.

        Args:
            radius: Sphere radius
            color: [r, g, b] color list
            resolution: Sphere mesh resolution

        Returns:
            Open3D TriangleMesh template
        """
        template = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        template.paint_uniform_color(color)
        template.compute_vertex_normals()
        return template

    def _hide_highlight(self, template):
        """Create a hidden copy of a highlight template moved off-screen."""
        hidden = o3d.geometry.TriangleMesh(template)
        hidden.translate([0, -10000, 0])
        return hidden

    def _update_highlight(self, template, position):
        """Create a highlight mesh at the given position.

        Args:
            template: Highlight template mesh
            position: [x, y, z] position to place highlight

        Returns:
            New mesh at position, or None if position is None
        """
        if position is None:
            return None
        highlight = o3d.geometry.TriangleMesh(template)
        highlight.translate(position)
        return highlight

    def _create_highlight_sphere(self):
        """Create the highlight sphere template and initial instance"""
        self._highlight_sphere_template = self._create_highlight_template(0.5, [1, 0, 0])
        self.highlight_sphere = self._hide_highlight(self._highlight_sphere_template)

    def update_highlight_position(self, idx: int):
        """Update the highlight sphere position by copying cached template"""
        point = self.level_data.get_point(idx)
        self.highlight_sphere = self._update_highlight(self._highlight_sphere_template, point)
        return self.highlight_sphere

    def hide_highlight(self):
        """Hide the highlight sphere by returning a copy moved off-screen"""
        self.highlight_sphere = self._hide_highlight(self._highlight_sphere_template)
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
        # Create centered box template
        size = self.SPAWN_HIGHLIGHT_SIZE
        self._spawn_highlight_template = o3d.geometry.TriangleMesh.create_box(
            width=size, height=size, depth=size
        )
        self._spawn_highlight_template.translate(np.array([-size / 2, -size / 2, -size / 2]))
        self._spawn_highlight_template.paint_uniform_color(self.SPAWN_HIGHLIGHT_COLOR)
        self._spawn_highlight_template.compute_vertex_normals()
        self.spawn_highlight = self._hide_highlight(self._spawn_highlight_template)

    def update_spawn_highlight_position(self, idx: int):
        """Update the spawn highlight cube position by copying cached template"""
        if self.spawn_data is None:
            return None
        pos = self.spawn_data.get_position(idx)
        self.spawn_highlight = self._update_highlight(self._spawn_highlight_template, pos)
        return self.spawn_highlight

    def hide_spawn_highlight(self):
        """Hide the spawn highlight by returning a copy moved off-screen"""
        self.spawn_highlight = self._hide_highlight(self._spawn_highlight_template)
        return self.spawn_highlight

    def _batch_create_spheres(self, positions, radius, resolution, color_func):
        """Create batched sphere mesh from positions array.

        Args:
            positions: Array of [x,y,z] positions for each sphere
            radius: Sphere radius
            resolution: Sphere mesh resolution (higher = smoother, more triangles)
            color_func: Function(index) -> [r,g,b] color for each sphere

        Returns:
            Open3D TriangleMesh with all spheres combined, or empty mesh if no positions
        """
        n = len(positions)
        if n == 0:
            return o3d.geometry.TriangleMesh()

        # Create template sphere at origin
        template = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
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

            # Set color from color function
            all_colors[v_start:v_start + n_verts] = color_func(i)

        # Create single mesh from arrays
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(all_verts)
        mesh.triangles = o3d.utility.Vector3iVector(all_tris)
        mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors)
        mesh.compute_vertex_normals()
        return mesh

    def _create_graph_vertices(self):
        """Create spheres for game graph vertices using batched template construction."""
        if self.graph_data is None or len(self.graph_data) == 0:
            self.graph_vertices = o3d.geometry.TriangleMesh()
            return

        positions = self.graph_data.positions
        inter_level_flags = self.graph_data.inter_level_flags

        def get_color(i):
            if i < len(inter_level_flags) and inter_level_flags[i]:
                return self.GRAPH_VERTEX_INTER_LEVEL_COLOR
            return self.GRAPH_VERTEX_COLOR

        self.graph_vertices = self._batch_create_spheres(
            positions, self.GRAPH_VERTEX_SIZE, resolution=8, color_func=get_color
        )

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
        self._graph_highlight_template = self._create_highlight_template(
            self.GRAPH_HIGHLIGHT_SIZE, self.GRAPH_HIGHLIGHT_COLOR
        )
        self.graph_highlight = self._hide_highlight(self._graph_highlight_template)

    def update_graph_highlight_position(self, idx: int):
        """Update the graph highlight sphere position by copying cached template"""
        if self.graph_data is None:
            return None
        pos = self.graph_data.get_position(idx)
        self.graph_highlight = self._update_highlight(self._graph_highlight_template, pos)
        return self.graph_highlight

    def hide_graph_highlight(self):
        """Hide the graph highlight by returning a copy moved off-screen"""
        self.graph_highlight = self._hide_highlight(self._graph_highlight_template)
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

    def _create_patrol_points(self):
        """Create spheres for patrol path points using batched template construction."""
        if self.patrol_data is None or len(self.patrol_data) == 0:
            self.patrol_points = o3d.geometry.TriangleMesh()
            return

        positions = self.patrol_data.positions
        self.patrol_points = self._batch_create_spheres(
            positions, self.PATROL_POINT_SIZE, resolution=6,
            color_func=lambda _: self.PATROL_POINT_COLOR
        )

    def _create_patrol_edges(self):
        """Create empty line set for patrol path edges (shown on selection only)."""
        self.patrol_edges = o3d.geometry.LineSet()

    def update_patrol_edges_for_patrol(self, patrol_name: str):
        """Update patrol edges to show only edges for the specified patrol.

        Args:
            patrol_name: Name of the patrol to show edges for

        Returns:
            Updated LineSet geometry
        """
        if self.patrol_data is None or len(self.patrol_data) == 0:
            self.patrol_edges = o3d.geometry.LineSet()
            return self.patrol_edges

        edges = self.patrol_data.get_patrol_edges(patrol_name)

        if len(edges) == 0:
            self.patrol_edges = o3d.geometry.LineSet()
            return self.patrol_edges

        positions = self.patrol_data.positions
        lines = np.array(edges, dtype=np.int32)
        line_colors = np.tile(self.PATROL_EDGE_COLOR, (len(edges), 1)).astype(np.float64)

        self.patrol_edges = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines),
        )
        self.patrol_edges.colors = o3d.utility.Vector3dVector(line_colors)
        return self.patrol_edges

    def hide_patrol_edges(self):
        """Hide patrol edges by returning an empty LineSet."""
        self.patrol_edges = o3d.geometry.LineSet()
        return self.patrol_edges

    def _create_patrol_highlight(self):
        """Create the patrol highlight sphere template and initial instance."""
        self._patrol_highlight_template = self._create_highlight_template(
            self.PATROL_HIGHLIGHT_SIZE, self.PATROL_HIGHLIGHT_COLOR
        )
        self.patrol_highlight = self._hide_highlight(self._patrol_highlight_template)

    def update_patrol_highlight_position(self, idx: int):
        """Update the patrol highlight sphere position by copying cached template."""
        if self.patrol_data is None:
            return None
        pos = self.patrol_data.get_position(idx)
        self.patrol_highlight = self._update_highlight(self._patrol_highlight_template, pos)
        return self.patrol_highlight

    def hide_patrol_highlight(self):
        """Hide the patrol highlight by returning a copy moved off-screen."""
        self.patrol_highlight = self._hide_highlight(self._patrol_highlight_template)
        return self.patrol_highlight
