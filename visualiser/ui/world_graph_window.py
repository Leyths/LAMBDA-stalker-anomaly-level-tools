"""
World Graph Window - displays all levels' game graph vertices in world space.
"""
from typing import Optional
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from core.world_graph_data import WorldGraphData


class WorldGraphWindow:
    """Standalone window displaying the world-wide game graph."""

    # Rendering constants
    VERTEX_RADIUS = 2.0
    VERTEX_RESOLUTION = 8
    HIGHLIGHT_RADIUS = 3.0
    HIGHLIGHT_COLOR = [1.0, 0.5, 0.0]  # Orange
    INTRA_EDGE_COLOR = [0.3, 0.6, 1.0]  # Blue
    INTER_EDGE_COLOR = [0.5, 0.0, 0.8]  # Purple

    # Picking parameters
    PICK_DISTANCE = 8.0
    PICK_RADIUS = 20.0
    SCREEN_THRESHOLD = 60.0

    # Top-down camera constants
    PAN_SENSITIVITY = 1.0  # World units per pixel (scaled by height)
    ZOOM_FACTOR = 1.05      # Multiply/divide height per scroll tick
    MIN_CAM_HEIGHT = 20.0   # Minimum camera height above the graph

    def __init__(self, all_spawn_path: str):
        self._all_spawn_path = all_spawn_path
        self.selected_vertex = None

        # Pan state
        self._pan_active = False
        self._pan_last_x = 0
        self._pan_last_y = 0

        # Camera state (top-down): eye is at (_cam_x, _cam_height, _cam_z)
        self._cam_x = 0.0
        self._cam_z = 0.0
        self._cam_height = 1000.0
        self._max_cam_height = 10000.0

        # Load data
        print("  Loading world graph data...", flush=True)
        self.data = WorldGraphData(all_spawn_path)

        if len(self.data) == 0:
            print("  Warning: No game graph vertices found!")
            return

        # Create geometry
        self._create_geometries()

        # Create window
        self.window = gui.Application.instance.create_window(
            "World Graph", 1400, 800
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)

        # Build side panel (using WidgetProxy so we can rebuild on selection change)
        self._build_panel()
        self._panel_wrapper = gui.WidgetProxy()
        self._panel_wrapper.set_widget(self.panel)
        self.window.add_child(self._panel_wrapper)

        # Setup scene
        self._setup_scene()
        self.window.add_child(self.scene)

        # Build spatial index
        self._build_kdtree()

        print(f"  World graph loaded: {len(self.data)} vertices, "
              f"{len(self.data.intra_level_edges)} intra-level edges, "
              f"{len(self.data.inter_level_edges)} inter-level edges, "
              f"{len(self.data.levels)} levels", flush=True)
        print("  Ready!", flush=True)

    def _create_geometries(self):
        """Create all Open3D geometry objects."""
        self._create_vertex_spheres()
        self._create_intra_edges()
        self._create_inter_edges()
        self._create_highlight()

    def _create_vertex_spheres(self):
        """Create batched sphere mesh for all vertices, colored by level."""
        positions = self.data.positions
        n = len(positions)

        if n == 0:
            self.vertex_mesh = o3d.geometry.TriangleMesh()
            return

        # Create template sphere
        template = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.VERTEX_RADIUS, resolution=self.VERTEX_RESOLUTION
        )
        template_verts = np.asarray(template.vertices)
        template_tris = np.asarray(template.triangles)
        n_verts = len(template_verts)
        n_tris = len(template_tris)

        # Pre-allocate arrays
        all_verts = np.empty((n * n_verts, 3), dtype=np.float64)
        all_tris = np.empty((n * n_tris, 3), dtype=np.int32)
        all_colors = np.empty((n * n_verts, 3), dtype=np.float64)

        for i in range(n):
            v_start = i * n_verts
            t_start = i * n_tris

            all_verts[v_start:v_start + n_verts] = template_verts + positions[i]
            all_tris[t_start:t_start + n_tris] = template_tris + v_start
            all_colors[v_start:v_start + n_verts] = self.data.get_vertex_color(i)

        self.vertex_mesh = o3d.geometry.TriangleMesh()
        self.vertex_mesh.vertices = o3d.utility.Vector3dVector(all_verts)
        self.vertex_mesh.triangles = o3d.utility.Vector3iVector(all_tris)
        self.vertex_mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors)
        self.vertex_mesh.compute_vertex_normals()

    def _create_line_set(self, edges, color):
        """Create a LineSet from edge pairs with uniform color."""
        if not edges:
            return o3d.geometry.LineSet()

        positions = self.data.positions
        lines = np.array(edges, dtype=np.int32)
        line_colors = np.tile(color, (len(edges), 1)).astype(np.float64)

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        return line_set

    def _create_intra_edges(self):
        """Create blue line set for intra-level edges."""
        self.intra_edges = self._create_line_set(
            self.data.intra_level_edges, self.INTRA_EDGE_COLOR
        )

    def _create_inter_edges(self):
        """Create purple line set for inter-level edges."""
        self.inter_edges = self._create_line_set(
            self.data.inter_level_edges, self.INTER_EDGE_COLOR
        )

    def _create_highlight(self):
        """Create highlight sphere template."""
        self._highlight_template = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.HIGHLIGHT_RADIUS, resolution=12
        )
        self._highlight_template.paint_uniform_color(self.HIGHLIGHT_COLOR)
        self._highlight_template.compute_vertex_normals()

        # Start hidden
        self.highlight_mesh = o3d.geometry.TriangleMesh(self._highlight_template)
        self.highlight_mesh.translate([0, -100000, 0])

    def _setup_scene(self):
        """Setup the 3D scene with all geometries."""
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.set_on_mouse(self._on_mouse)
        self.scene.scene.show_axes(False)
        self.scene.scene.show_skybox(False)

        # Vertex spheres
        vertex_mat = rendering.MaterialRecord()
        vertex_mat.shader = "defaultLit"
        self.scene.scene.add_geometry("vertices", self.vertex_mesh, vertex_mat)

        # Intra-level edges (blue)
        intra_mat = rendering.MaterialRecord()
        intra_mat.shader = "unlitLine"
        intra_mat.line_width = 2
        self.scene.scene.add_geometry("intra_edges", self.intra_edges, intra_mat)

        # Inter-level edges (purple)
        inter_mat = rendering.MaterialRecord()
        inter_mat.shader = "unlitLine"
        inter_mat.line_width = 3
        self.scene.scene.add_geometry("inter_edges", self.inter_edges, inter_mat)

        # Highlight
        highlight_mat = rendering.MaterialRecord()
        highlight_mat.shader = "defaultLit"
        self.scene.scene.add_geometry("highlight", self.highlight_mesh, highlight_mat)

        # Setup top-down camera looking straight down at the world
        positions = self.data.positions
        min_bound = positions.min(axis=0)
        max_bound = positions.max(axis=0)
        center = (min_bound + max_bound) / 2.0
        extent = max_bound - min_bound

        self._cam_x = center[0]
        self._cam_z = center[2]
        # Height so the full extent fits in view with some margin
        self._cam_height = max(extent[0], abs(extent[2])) * 1.0
        self._max_cam_height = self._cam_height * 2.0

        self._apply_top_down_camera()

    def _build_kdtree(self):
        """Build KDTree spatial index for picking."""
        positions = self.data.positions
        if len(positions) == 0:
            self._kdtree = None
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        self._kdtree = o3d.geometry.KDTreeFlann(pcd)

    def _build_panel(self):
        """Build the side panel."""
        self.panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        # Title
        title = gui.Label("World Graph")
        title.text_color = gui.Color(1, 1, 0)
        self.panel.add_child(title)
        self.panel.add_fixed(10)

        # Controls
        controls_text = (
            "Controls:\n"
            "- Left Click+Drag: Pan\n"
            "- Scroll: Zoom in/out\n"
            "- Ctrl+Click: Select vertex\n"
            "- Space: Focus on selection\n"
        )
        controls_label = gui.Label(controls_text)
        controls_label.text_color = gui.Color(0.7, 0.7, 0.7)
        self.panel.add_child(controls_label)
        self.panel.add_fixed(10)

        # Stats
        n_verts = len(self.data)
        n_intra = len(self.data.intra_level_edges)
        n_inter = len(self.data.inter_level_edges)
        n_levels = len(self.data.levels)

        stats_text = (
            f"Vertices: {n_verts}\n"
            f"Intra-level edges: {n_intra}\n"
            f"Inter-level edges: {n_inter}\n"
            f"Levels: {n_levels}"
        )
        stats_label = gui.Label(stats_text)
        stats_label.text_color = gui.Color(0.8, 0.8, 0.8)
        self.panel.add_child(stats_label)
        self.panel.add_fixed(15)

        # Info area
        if self.selected_vertex is not None:
            vertex = self.data.get_vertex(self.selected_vertex)
            pos = self.data.get_position(self.selected_vertex)
            edges_info = self.data.get_edges_info(self.selected_vertex)

            if vertex is not None and pos is not None:
                level = self.data.levels.get(vertex.level_id)
                level_name = level.name if level else f"Unknown ({vertex.level_id})"

                lp = vertex.local_point
                info_text = (
                    f"GVID: {vertex.vertex_id}\n"
                    f"Level: {level_name}\n"
                    f"World: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
                    f"Local: ({lp[0]:.1f}, {lp[1]:.1f}, {lp[2]:.1f})\n"
                    f"Edges: {len(edges_info)}"
                )
                info_label = gui.Label(info_text)
                info_label.text_color = gui.Color(1, 1, 1)
                self.panel.add_child(info_label)
                self.panel.add_fixed(5)

                for edge_info in edges_info:
                    target_id = edge_info['target_vertex_id']
                    distance = edge_info['distance']
                    is_inter = edge_info['is_inter_level']
                    target_name = edge_info['level_name']
                    edge_type = "INTER" if is_inter else "intra"
                    edge_label = gui.Label(f"  -> GVID {target_id} ({target_name}) d={distance:.0f} [{edge_type}]")
                    if not is_inter:
                        edge_label.text_color = gui.Color(
                            self.INTRA_EDGE_COLOR[0], self.INTRA_EDGE_COLOR[1], self.INTRA_EDGE_COLOR[2]
                        )
                    else:
                        edge_label.text_color = gui.Color(
                            self.INTER_EDGE_COLOR[0], self.INTER_EDGE_COLOR[1], self.INTER_EDGE_COLOR[2]
                        )
                    self.panel.add_child(edge_label)
            else:
                info_label = gui.Label("Ctrl+Click a vertex to inspect it")
                info_label.text_color = gui.Color(1, 1, 1)
                self.panel.add_child(info_label)
        else:
            info_label = gui.Label("Ctrl+Click a vertex to inspect it")
            info_label.text_color = gui.Color(1, 1, 1)
            self.panel.add_child(info_label)
        self.panel.add_fixed(15)

        # Edge color legend
        edge_legend_title = gui.Label("Edge Colors:")
        edge_legend_title.text_color = gui.Color(0.8, 0.8, 0.8)
        self.panel.add_child(edge_legend_title)
        self.panel.add_fixed(5)

        intra_label = gui.Label("  Intra-level (same level)")
        intra_label.text_color = gui.Color(
            self.INTRA_EDGE_COLOR[0], self.INTRA_EDGE_COLOR[1], self.INTRA_EDGE_COLOR[2]
        )
        self.panel.add_child(intra_label)

        inter_label = gui.Label("  Inter-level (between levels)")
        inter_label.text_color = gui.Color(
            self.INTER_EDGE_COLOR[0], self.INTER_EDGE_COLOR[1], self.INTER_EDGE_COLOR[2]
        )
        self.panel.add_child(inter_label)
        self.panel.add_fixed(15)

        # Level color legend
        legend_title = gui.Label("Level Colors:")
        legend_title.text_color = gui.Color(0.8, 0.8, 0.8)
        self.panel.add_child(legend_title)
        self.panel.add_fixed(5)

        levels = self.data.levels
        colors = self.data.level_colors
        for lid in sorted(levels.keys()):
            level = levels[lid]
            color = colors.get(lid, [0.5, 0.5, 0.5])
            label = gui.Label(f"  {level.name}")
            label.text_color = gui.Color(color[0], color[1], color[2])
            self.panel.add_child(label)

    def _on_layout(self, layout_context):
        """Handle window layout."""
        r = self.window.content_rect

        panel_width = 350
        panel_rect = gui.Rect(r.width - panel_width - 10, 10, panel_width, r.height - 20)
        self._panel_wrapper.frame = panel_rect

        self.scene.frame = gui.Rect(0, 0, r.width - panel_width - 20, r.height)

    def _on_mouse(self, event):
        """Handle mouse events for top-down pan/zoom camera."""
        # Block middle mouse and right mouse (no rotation)
        if event.is_button_down(gui.MouseButton.MIDDLE) or \
           event.is_button_down(gui.MouseButton.RIGHT):
            return gui.SceneWidget.EventCallbackResult.CONSUMED

        # Ctrl+Click to pick
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and \
           event.is_button_down(gui.MouseButton.LEFT) and \
           event.is_modifier_down(gui.KeyModifier.CTRL):
            self._pan_active = False
            self._pick_vertex(event.x, event.y)
            return gui.SceneWidget.EventCallbackResult.CONSUMED

        # Left click - start pan
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and \
           event.is_button_down(gui.MouseButton.LEFT):
            self._pan_active = True
            self._pan_last_x = event.x
            self._pan_last_y = event.y
            return gui.SceneWidget.EventCallbackResult.CONSUMED

        # Drag - pan
        if self._pan_active and event.type in (gui.MouseEvent.Type.DRAG, gui.MouseEvent.Type.MOVE):
            dx = event.x - self._pan_last_x
            dy = event.y - self._pan_last_y
            self._pan_last_x = event.x
            self._pan_last_y = event.y
            if abs(dx) > 0 or abs(dy) > 0:
                self._apply_pan(dx, dy)
            return gui.SceneWidget.EventCallbackResult.CONSUMED

        # Button release
        if event.type == gui.MouseEvent.Type.BUTTON_UP:
            if self._pan_active:
                self._pan_active = False
                return gui.SceneWidget.EventCallbackResult.CONSUMED

        # Scroll - zoom
        if event.type == gui.MouseEvent.Type.WHEEL:
            self._apply_zoom(event.wheel_dy)
            return gui.SceneWidget.EventCallbackResult.CONSUMED

        return gui.SceneWidget.EventCallbackResult.IGNORED

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.KeyName.SPACE and self.selected_vertex is not None:
                self._focus_on_vertex(self.selected_vertex)
                return True
        return False

    def _apply_pan(self, dx, dy):
        """Pan the top-down camera by pixel delta."""
        # Scale pan speed by camera height so it feels consistent at any zoom
        viewport_height = max(self.scene.frame.height, 1)
        scale = self._cam_height / viewport_height * 2.0
        self._cam_x -= dx * scale
        self._cam_z -= dy * scale  # screen Y maps to world Z (mirrored)
        self._apply_top_down_camera()

    def _apply_zoom(self, wheel_dy):
        """Zoom by adjusting camera height. Positive wheel_dy = zoom in."""
        if wheel_dy > 0:
            self._cam_height /= self.ZOOM_FACTOR
        elif wheel_dy < 0:
            self._cam_height *= self.ZOOM_FACTOR

        # Clamp
        self._cam_height = np.clip(self._cam_height, self.MIN_CAM_HEIGHT, self._max_cam_height)
        self._apply_top_down_camera()

    def _apply_top_down_camera(self):
        """Set the camera to look straight down at (_cam_x, 0, _cam_z) from _cam_height."""
        target = np.array([self._cam_x, 0.0, self._cam_z])
        eye = np.array([self._cam_x, self._cam_height, self._cam_z])
        # Up direction is -Z in world (so north on screen = -Z in world coords)
        self.scene.look_at(target, eye, [0.0, 0.0, -1.0])

    def _compute_pick_ray(self, screen_x, screen_y):
        """Compute pick ray from screen coordinates."""
        view_matrix = self.scene.scene.camera.get_view_matrix()
        proj_matrix = self.scene.scene.camera.get_projection_matrix()

        viewport_width = self.scene.frame.width
        viewport_height = self.scene.frame.height

        if viewport_width == 0 or viewport_height == 0:
            return None

        ndc_x = (2.0 * screen_x) / viewport_width - 1.0
        ndc_y = 1.0 - (2.0 * screen_y) / viewport_height

        near_ndc = np.array([ndc_x, ndc_y, 0.0, 1.0])
        far_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0])

        view_mat = np.array(view_matrix).reshape(4, 4)
        proj_mat = np.array(proj_matrix).reshape(4, 4)
        vp_matrix = proj_mat @ view_mat

        try:
            inv_vp_matrix = np.linalg.inv(vp_matrix)
        except Exception:
            return None

        near_world = inv_vp_matrix @ near_ndc
        far_world = inv_vp_matrix @ far_ndc

        if abs(near_world[3]) < 1e-6 or abs(far_world[3]) < 1e-6:
            return None

        near_world = near_world[:3] / near_world[3]
        far_world = far_world[:3] / far_world[3]

        ray_origin = near_world
        ray_direction = far_world - near_world
        ray_length = np.linalg.norm(ray_direction)

        if ray_length < 1e-6:
            return None

        ray_direction = ray_direction / ray_length
        return ray_origin, ray_direction, ray_length

    def _pick_vertex(self, screen_x, screen_y):
        """Pick the nearest vertex to the click position."""
        if self._kdtree is None:
            return

        ray_data = self._compute_pick_ray(screen_x, screen_y)
        idx = None

        if ray_data is not None:
            ray_origin, ray_direction, ray_length = ray_data
            idx, _ = self._pick_along_ray(ray_origin, ray_direction, ray_length)

        if idx is None:
            # Fallback: screen-space distance
            idx = self._pick_screen_distance(screen_x, screen_y)

        if idx is not None:
            self._select_vertex(idx)

    def _pick_along_ray(self, ray_origin, ray_direction, ray_length):
        """Pick vertex along a ray using KDTree sampling."""
        closest_idx = None
        closest_distance = float('inf')
        checked = set()
        positions = self.data.positions

        for t in np.linspace(0, ray_length, 12):
            sample = ray_origin + t * ray_direction
            [k, idx, _] = self._kdtree.search_radius_vector_3d(sample, self.PICK_RADIUS)

            for i in range(k):
                point_idx = idx[i]
                if point_idx in checked:
                    continue
                checked.add(point_idx)

                point = positions[point_idx]
                to_point = point - ray_origin
                proj_len = np.dot(to_point, ray_direction)

                if proj_len > 0:
                    closest_on_ray = ray_origin + proj_len * ray_direction
                    dist = np.linalg.norm(point - closest_on_ray)

                    if dist < closest_distance and dist < self.PICK_DISTANCE:
                        closest_distance = dist
                        closest_idx = point_idx

        return closest_idx, closest_distance

    def _pick_screen_distance(self, screen_x, screen_y):
        """Fallback picking using screen-space distance."""
        viewport_width = self.scene.frame.width
        viewport_height = self.scene.frame.height
        positions = self.data.positions

        if viewport_width == 0 or viewport_height == 0 or len(positions) == 0:
            return None

        view_mat = np.array(self.scene.scene.camera.get_view_matrix()).reshape(4, 4)
        proj_mat = np.array(self.scene.scene.camera.get_projection_matrix()).reshape(4, 4)
        vp_matrix = proj_mat @ view_mat

        pos_h = np.hstack((positions, np.ones((len(positions), 1))))
        clip_pos = pos_h @ vp_matrix.T
        w = clip_pos[:, 3]

        valid_mask = w > 1e-6
        dist_sq = np.full(len(positions), np.inf)

        if np.any(valid_mask):
            valid_clip = clip_pos[valid_mask]
            valid_w = w[valid_mask]
            ndc = valid_clip[:, :3] / valid_w[:, np.newaxis]
            depth_mask = (ndc[:, 2] >= -1) & (ndc[:, 2] <= 1)

            final_indices = np.where(valid_mask)[0][depth_mask]
            valid_ndc = ndc[depth_mask]

            sx = (valid_ndc[:, 0] + 1.0) * viewport_width / 2.0
            sy = (1.0 - valid_ndc[:, 1]) * viewport_height / 2.0

            dx = sx - screen_x
            dy = sy - screen_y
            dist_sq[final_indices] = dx * dx + dy * dy

        closest_idx = np.argmin(dist_sq)
        if dist_sq[closest_idx] < self.SCREEN_THRESHOLD * self.SCREEN_THRESHOLD:
            return int(closest_idx)
        return None

    def _select_vertex(self, idx):
        """Select a vertex and update display."""
        self.selected_vertex = idx

        # Update highlight (don't move camera on click - just highlight)
        self.scene.scene.remove_geometry("highlight")
        pos = self.data.get_position(idx)
        if pos is not None:
            self.highlight_mesh = o3d.geometry.TriangleMesh(self._highlight_template)
            self.highlight_mesh.translate(pos)
        else:
            self.highlight_mesh = o3d.geometry.TriangleMesh(self._highlight_template)
            self.highlight_mesh.translate([0, -100000, 0])

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.scene.scene.add_geometry("highlight", self.highlight_mesh, mat)

        # Rebuild panel to show updated info without overlap
        self._build_panel()
        self._panel_wrapper.set_widget(self.panel)
        self.window.set_needs_layout()

        self.scene.force_redraw()

    def _focus_on_vertex(self, idx):
        """Pan camera to center on a vertex."""
        pos = self.data.get_position(idx)
        if pos is not None:
            self._cam_x = pos[0]
            self._cam_z = pos[2]
            self._apply_top_down_camera()
