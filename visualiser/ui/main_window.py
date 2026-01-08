"""
Main application window for read-only node graph visualization.
"""
import os
from typing import Optional
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from core.data_loader import LevelData, SpawnData, GraphData
from core.geometry import GeometryManager
from .dialogs import DialogFactory
from .control_panel import ControlPanel


class NodeInspectorApp:
    """Main application for node graph inspection (read-only)."""

    def __init__(self, level_file: str, level_id: Optional[int] = None, all_spawn_path: Optional[str] = None):
        # Load data (pass all_spawn_path and level_id for cross-table loading)
        self.level_data = LevelData(level_file, all_spawn_path, level_id)

        # Load spawn data if level_id and all_spawn_path are provided
        self.spawn_data: Optional[SpawnData] = None
        self.graph_data: Optional[GraphData] = None
        if level_id is not None and all_spawn_path and os.path.exists(all_spawn_path):
            self.spawn_data = SpawnData(all_spawn_path, level_id)
            print(f"Loaded {len(self.spawn_data)} spawn objects for this level")
            self.graph_data = GraphData(all_spawn_path, level_id)
            print(f"Loaded {len(self.graph_data)} game graph vertices for this level")

        self.geometry_manager = GeometryManager(self.level_data, self.spawn_data, self.graph_data)

        # State
        self.selected_node = None
        self.selected_spawn = None
        self.selected_graph = None

        # Create window
        self.window = gui.Application.instance.create_window(
            "Leyths' Level Vertex Graph Inspector", 1400, 800
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)

        # Create scene
        self._setup_scene()

        # Create UI
        self.control_panel = ControlPanel(
            self.level_data,
            self._on_node_selected,
            self._on_coordinate_search,
            spawn_data=self.spawn_data,
            graph_data=self.graph_data,
            on_panel_rebuild=self._on_panel_rebuild,
            on_spawn_selected=self._on_spawn_selected,
            on_graph_selected=self._on_graph_selected
        )
        self.control_panel.set_window(self.window)
        self.window.add_child(self.control_panel.panel)

        # Initialize - select node 0 and focus camera on it
        self.inspect_node(0, reset_camera=True)

    def _setup_scene(self):
        """Setup the 3D scene."""
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.set_on_mouse(self._on_mouse)

        # Disable coordinate frame
        self.scene.scene.show_axes(False)
        self.scene.scene.show_skybox(False)

        self.window.add_child(self.scene)

        # Add geometries
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 5
        self.scene.scene.add_geometry(
            "point_cloud",
            self.geometry_manager.point_cloud,
            mat
        )

        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 1
        self.scene.scene.add_geometry(
            "lines",
            self.geometry_manager.line_set,
            line_mat
        )

        highlight_mat = rendering.MaterialRecord()
        highlight_mat.shader = "defaultLit"
        self.scene.scene.add_geometry(
            "highlight",
            self.geometry_manager.highlight_sphere,
            highlight_mat
        )

        # Add spawn geometries if available
        if self.spawn_data is not None and len(self.spawn_data) > 0:
            spawn_mat = rendering.MaterialRecord()
            spawn_mat.shader = "defaultLit"
            self.scene.scene.add_geometry(
                "spawn_points",
                self.geometry_manager.spawn_points,
                spawn_mat
            )

            spawn_highlight_mat = rendering.MaterialRecord()
            spawn_highlight_mat.shader = "defaultLitTransparency"
            spawn_highlight_mat.base_color = [1.0, 1.0, 0.0, 0.3]  # Yellow with 30% opacity
            self.scene.scene.add_geometry(
                "spawn_highlight",
                self.geometry_manager.spawn_highlight,
                spawn_highlight_mat
            )

            # Add space_restrictor shapes with semi-transparency
            if self.geometry_manager.space_restrictor_shapes is not None:
                shape_mat = rendering.MaterialRecord()
                shape_mat.shader = "defaultLitTransparency"
                shape_mat.base_color = [0.0, 1.0, 0.5, 0.3]  # Cyan-green with 30% opacity
                self.scene.scene.add_geometry(
                    "space_restrictor_shapes",
                    self.geometry_manager.space_restrictor_shapes,
                    shape_mat
                )

        # Add game graph geometries if available
        if self.graph_data is not None and len(self.graph_data) > 0:
            # Add graph vertices
            graph_vertex_mat = rendering.MaterialRecord()
            graph_vertex_mat.shader = "defaultLit"
            self.scene.scene.add_geometry(
                "graph_vertices",
                self.geometry_manager.graph_vertices,
                graph_vertex_mat
            )

            # Add graph edges
            graph_edge_mat = rendering.MaterialRecord()
            graph_edge_mat.shader = "unlitLine"
            graph_edge_mat.line_width = 2
            self.scene.scene.add_geometry(
                "graph_edges",
                self.geometry_manager.graph_edges,
                graph_edge_mat
            )

            # Add graph highlight
            graph_highlight_mat = rendering.MaterialRecord()
            graph_highlight_mat.shader = "defaultLit"
            self.scene.scene.add_geometry(
                "graph_highlight",
                self.geometry_manager.graph_highlight,
                graph_highlight_mat
            )

        # Setup camera
        bounds = self.scene.scene.bounding_box
        self.scene.setup_camera(60, bounds, bounds.get_center())

    def _on_layout(self, layout_context):
        """Handle window layout."""
        r = self.window.content_rect

        # Control panel on the right
        panel_width = 350
        self.control_panel.set_frame(
            r.width - panel_width - 10,
            10,
            panel_width,
            r.height - 20
        )

        # Scene takes the rest
        self.scene.frame = gui.Rect(0, 0, r.width - panel_width - 20, r.height)

    def _on_panel_rebuild(self, old_panel, new_panel):
        """Handle control panel rebuild - swap old panel for new one."""
        # Open3D doesn't support removing children, so we hide the old panel
        # by moving it off-screen
        old_panel.frame = gui.Rect(-10000, -10000, 1, 1)

        self.window.add_child(new_panel)
        # Trigger a layout update
        self.window.set_needs_layout()

    def _on_mouse(self, event):
        """Handle mouse events."""
        # Block middle mouse button
        if event.is_button_down(gui.MouseButton.MIDDLE) or \
           event.type == gui.MouseEvent.Type.BUTTON_DOWN and \
           event.is_button_down(gui.MouseButton.MIDDLE):
            return gui.SceneWidget.EventCallbackResult.HANDLED

        # Ctrl+Click to pick node, spawn, or graph vertex (moves camera but keeps viewing angle/distance)
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and \
           event.is_button_down(gui.MouseButton.LEFT) and \
           event.is_modifier_down(gui.KeyModifier.CTRL):

            # Pick graph vertices, spawns, and level vertices
            picked_graph_idx, graph_dist = self._pick_graph_at_screen_pos(event.x, event.y)
            picked_spawn_idx, spawn_dist = self._pick_spawn_at_screen_pos(event.x, event.y)
            picked_node_idx, node_dist = self._pick_node_at_screen_pos_with_dist(event.x, event.y)

            # Priority thresholds - graph and spawn points get priority over dense nav mesh
            graph_priority_threshold = 40.0  # Screen pixels
            spawn_priority_threshold = 50.0  # Screen pixels

            # Graph vertices have highest priority (they're larger and more meaningful)
            if picked_graph_idx is not None and graph_dist < graph_priority_threshold:
                self.inspect_graph(picked_graph_idx, move_camera=True)
            # Then spawn points
            elif picked_spawn_idx is not None and spawn_dist < spawn_priority_threshold:
                self.inspect_spawn(picked_spawn_idx, move_camera=True)
            # Then compare distances between all candidates
            else:
                candidates = []
                if picked_graph_idx is not None:
                    candidates.append(('graph', picked_graph_idx, graph_dist))
                if picked_spawn_idx is not None:
                    candidates.append(('spawn', picked_spawn_idx, spawn_dist))
                if picked_node_idx is not None:
                    candidates.append(('node', picked_node_idx, node_dist))

                if candidates:
                    # Sort by distance and pick closest
                    candidates.sort(key=lambda x: x[2])
                    best_type, best_idx, _ = candidates[0]
                    if best_type == 'graph':
                        self.inspect_graph(best_idx, move_camera=True)
                    elif best_type == 'spawn':
                        self.inspect_spawn(best_idx, move_camera=True)
                    else:
                        self.inspect_node(best_idx, move_camera=True)

            return gui.SceneWidget.EventCallbackResult.HANDLED

        return gui.SceneWidget.EventCallbackResult.IGNORED

    def _pick_node_at_screen_pos(self, screen_x, screen_y):
        """Pick a node at screen coordinates using ray casting."""
        view_matrix = self.scene.scene.camera.get_view_matrix()
        proj_matrix = self.scene.scene.camera.get_projection_matrix()

        viewport_width = self.scene.frame.width
        viewport_height = self.scene.frame.height

        if viewport_width == 0 or viewport_height == 0:
            return None

        # Convert to NDC
        ndc_x = (2.0 * screen_x) / viewport_width - 1.0
        ndc_y = 1.0 - (2.0 * screen_y) / viewport_height

        near_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])
        far_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0])

        view_mat = np.array(view_matrix).reshape(4, 4)
        proj_mat = np.array(proj_matrix).reshape(4, 4)
        vp_matrix = proj_mat @ view_mat

        try:
            inv_vp_matrix = np.linalg.inv(vp_matrix)
        except:
            return self._pick_node_screen_distance(screen_x, screen_y)

        near_world = inv_vp_matrix @ near_ndc
        far_world = inv_vp_matrix @ far_ndc

        if abs(near_world[3]) < 1e-6 or abs(far_world[3]) < 1e-6:
            return self._pick_node_screen_distance(screen_x, screen_y)

        near_world = near_world[:3] / near_world[3]
        far_world = far_world[:3] / far_world[3]

        ray_origin = near_world
        ray_direction = far_world - near_world
        ray_length = np.linalg.norm(ray_direction)

        if ray_length < 1e-6:
            return self._pick_node_screen_distance(screen_x, screen_y)

        ray_direction = ray_direction / ray_length

        # Find closest node to ray
        closest_idx = None
        closest_distance = float('inf')
        distance_threshold = 2.0

        for i in range(len(self.level_data)):
            point = self.level_data.get_point(i)
            to_point = point - ray_origin
            projection_length = np.dot(to_point, ray_direction)

            if projection_length > 0:
                closest_on_ray = ray_origin + projection_length * ray_direction
                distance_to_ray = np.linalg.norm(point - closest_on_ray)

                if distance_to_ray < closest_distance and distance_to_ray < distance_threshold:
                    closest_distance = distance_to_ray
                    closest_idx = i

        return closest_idx

    def _pick_node_at_screen_pos_with_dist(self, screen_x, screen_y):
        """Pick a node at screen coordinates and return (index, distance) tuple."""
        view_matrix = self.scene.scene.camera.get_view_matrix()
        proj_matrix = self.scene.scene.camera.get_projection_matrix()

        viewport_width = self.scene.frame.width
        viewport_height = self.scene.frame.height

        if viewport_width == 0 or viewport_height == 0:
            return None, float('inf')

        # Convert to NDC
        ndc_x = (2.0 * screen_x) / viewport_width - 1.0
        ndc_y = 1.0 - (2.0 * screen_y) / viewport_height

        near_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])
        far_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0])

        view_mat = np.array(view_matrix).reshape(4, 4)
        proj_mat = np.array(proj_matrix).reshape(4, 4)
        vp_matrix = proj_mat @ view_mat

        try:
            inv_vp_matrix = np.linalg.inv(vp_matrix)
        except Exception:
            idx = self._pick_node_screen_distance(screen_x, screen_y)
            return idx, float('inf') if idx is None else 0.0

        near_world = inv_vp_matrix @ near_ndc
        far_world = inv_vp_matrix @ far_ndc

        if abs(near_world[3]) < 1e-6 or abs(far_world[3]) < 1e-6:
            idx = self._pick_node_screen_distance(screen_x, screen_y)
            return idx, float('inf') if idx is None else 0.0

        near_world = near_world[:3] / near_world[3]
        far_world = far_world[:3] / far_world[3]

        ray_origin = near_world
        ray_direction = far_world - near_world
        ray_length = np.linalg.norm(ray_direction)

        if ray_length < 1e-6:
            idx = self._pick_node_screen_distance(screen_x, screen_y)
            return idx, float('inf') if idx is None else 0.0

        ray_direction = ray_direction / ray_length

        # Find closest node to ray
        closest_idx = None
        closest_distance = float('inf')
        distance_threshold = 2.0

        for i in range(len(self.level_data)):
            point = self.level_data.get_point(i)
            to_point = point - ray_origin
            projection_length = np.dot(to_point, ray_direction)

            if projection_length > 0:
                closest_on_ray = ray_origin + projection_length * ray_direction
                distance_to_ray = np.linalg.norm(point - closest_on_ray)

                if distance_to_ray < closest_distance and distance_to_ray < distance_threshold:
                    closest_distance = distance_to_ray
                    closest_idx = i

        return closest_idx, closest_distance

    def _pick_spawn_at_screen_pos(self, screen_x, screen_y):
        """Pick a spawn object at screen coordinates using ray casting.

        Returns:
            (spawn_index, distance) tuple
        """
        if self.spawn_data is None or len(self.spawn_data) == 0:
            return None, float('inf')

        view_matrix = self.scene.scene.camera.get_view_matrix()
        proj_matrix = self.scene.scene.camera.get_projection_matrix()

        viewport_width = self.scene.frame.width
        viewport_height = self.scene.frame.height

        if viewport_width == 0 or viewport_height == 0:
            return None, float('inf')

        # Convert to NDC
        ndc_x = (2.0 * screen_x) / viewport_width - 1.0
        ndc_y = 1.0 - (2.0 * screen_y) / viewport_height

        near_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])
        far_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0])

        view_mat = np.array(view_matrix).reshape(4, 4)
        proj_mat = np.array(proj_matrix).reshape(4, 4)
        vp_matrix = proj_mat @ view_mat

        try:
            inv_vp_matrix = np.linalg.inv(vp_matrix)
        except:
            return self._pick_spawn_screen_distance(screen_x, screen_y)

        near_world = inv_vp_matrix @ near_ndc
        far_world = inv_vp_matrix @ far_ndc

        if abs(near_world[3]) < 1e-6 or abs(far_world[3]) < 1e-6:
            return self._pick_spawn_screen_distance(screen_x, screen_y)

        near_world = near_world[:3] / near_world[3]
        far_world = far_world[:3] / far_world[3]

        ray_origin = near_world
        ray_direction = far_world - near_world
        ray_length = np.linalg.norm(ray_direction)

        if ray_length < 1e-6:
            return self._pick_spawn_screen_distance(screen_x, screen_y)

        ray_direction = ray_direction / ray_length

        # Find closest spawn to ray
        # Use larger threshold than vertices since spawns are less dense
        closest_idx = None
        closest_distance = float('inf')
        distance_threshold = 3.0

        for i in range(len(self.spawn_data)):
            point = self.spawn_data.get_position(i)
            if point is None:
                continue
            to_point = point - ray_origin
            projection_length = np.dot(to_point, ray_direction)

            if projection_length > 0:
                closest_on_ray = ray_origin + projection_length * ray_direction
                distance_to_ray = np.linalg.norm(point - closest_on_ray)

                if distance_to_ray < closest_distance and distance_to_ray < distance_threshold:
                    closest_distance = distance_to_ray
                    closest_idx = i

        return closest_idx, closest_distance

    def _pick_node_screen_distance(self, screen_x, screen_y):
        """Fallback picking using screen-space distance."""
        view_matrix = self.scene.scene.camera.get_view_matrix()
        proj_matrix = self.scene.scene.camera.get_projection_matrix()

        viewport_width = self.scene.frame.width
        viewport_height = self.scene.frame.height

        if viewport_width == 0 or viewport_height == 0:
            return None

        view_mat = np.array(view_matrix).reshape(4, 4)
        proj_mat = np.array(proj_matrix).reshape(4, 4)
        vp_matrix = proj_mat @ view_mat

        closest_idx = None
        closest_screen_dist = float('inf')
        max_screen_dist = 30.0

        for i in range(len(self.level_data)):
            point = self.level_data.get_point(i)
            point_h = np.array([point[0], point[1], point[2], 1.0])
            clip_pos = vp_matrix @ point_h

            if abs(clip_pos[3]) < 1e-6 or clip_pos[3] < 0:
                continue

            ndc = clip_pos[:3] / clip_pos[3]

            if ndc[2] < -1 or ndc[2] > 1:
                continue

            screen_pos_x = (ndc[0] + 1.0) * viewport_width / 2.0
            screen_pos_y = (1.0 - ndc[1]) * viewport_height / 2.0

            dx = screen_pos_x - screen_x
            dy = screen_pos_y - screen_y
            screen_dist = np.sqrt(dx*dx + dy*dy)

            if screen_dist < closest_screen_dist and screen_dist < max_screen_dist:
                closest_screen_dist = screen_dist
                closest_idx = i

        return closest_idx

    def _pick_spawn_screen_distance(self, screen_x, screen_y):
        """Fallback spawn picking using screen-space distance."""
        if self.spawn_data is None or len(self.spawn_data) == 0:
            return None, float('inf')

        view_matrix = self.scene.scene.camera.get_view_matrix()
        proj_matrix = self.scene.scene.camera.get_projection_matrix()

        viewport_width = self.scene.frame.width
        viewport_height = self.scene.frame.height

        if viewport_width == 0 or viewport_height == 0:
            return None, float('inf')

        view_mat = np.array(view_matrix).reshape(4, 4)
        proj_mat = np.array(proj_matrix).reshape(4, 4)
        vp_matrix = proj_mat @ view_mat

        closest_idx = None
        closest_screen_dist = float('inf')
        max_screen_dist = 50.0  # Larger threshold for spawns

        for i in range(len(self.spawn_data)):
            point = self.spawn_data.get_position(i)
            if point is None:
                continue
            point_h = np.array([point[0], point[1], point[2], 1.0])
            clip_pos = vp_matrix @ point_h

            if abs(clip_pos[3]) < 1e-6 or clip_pos[3] < 0:
                continue

            ndc = clip_pos[:3] / clip_pos[3]

            if ndc[2] < -1 or ndc[2] > 1:
                continue

            screen_pos_x = (ndc[0] + 1.0) * viewport_width / 2.0
            screen_pos_y = (1.0 - ndc[1]) * viewport_height / 2.0

            dx = screen_pos_x - screen_x
            dy = screen_pos_y - screen_y
            screen_dist = np.sqrt(dx*dx + dy*dy)

            if screen_dist < closest_screen_dist and screen_dist < max_screen_dist:
                closest_screen_dist = screen_dist
                closest_idx = i

        return closest_idx, closest_screen_dist

    def _pick_graph_at_screen_pos(self, screen_x, screen_y):
        """Pick a game graph vertex at screen coordinates using ray casting.

        Returns:
            (graph_index, distance) tuple
        """
        if self.graph_data is None or len(self.graph_data) == 0:
            return None, float('inf')

        view_matrix = self.scene.scene.camera.get_view_matrix()
        proj_matrix = self.scene.scene.camera.get_projection_matrix()

        viewport_width = self.scene.frame.width
        viewport_height = self.scene.frame.height

        if viewport_width == 0 or viewport_height == 0:
            return None, float('inf')

        # Convert to NDC
        ndc_x = (2.0 * screen_x) / viewport_width - 1.0
        ndc_y = 1.0 - (2.0 * screen_y) / viewport_height

        near_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])
        far_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0])

        view_mat = np.array(view_matrix).reshape(4, 4)
        proj_mat = np.array(proj_matrix).reshape(4, 4)
        vp_matrix = proj_mat @ view_mat

        try:
            inv_vp_matrix = np.linalg.inv(vp_matrix)
        except Exception:
            return self._pick_graph_screen_distance(screen_x, screen_y)

        near_world = inv_vp_matrix @ near_ndc
        far_world = inv_vp_matrix @ far_ndc

        if abs(near_world[3]) < 1e-6 or abs(far_world[3]) < 1e-6:
            return self._pick_graph_screen_distance(screen_x, screen_y)

        near_world = near_world[:3] / near_world[3]
        far_world = far_world[:3] / far_world[3]

        ray_origin = near_world
        ray_direction = far_world - near_world
        ray_length = np.linalg.norm(ray_direction)

        if ray_length < 1e-6:
            return self._pick_graph_screen_distance(screen_x, screen_y)

        ray_direction = ray_direction / ray_length

        # Find closest graph vertex to ray
        # Use larger threshold than level vertices since graph vertices are larger
        closest_idx = None
        closest_distance = float('inf')
        distance_threshold = 4.0  # Larger threshold for graph vertices

        for i in range(len(self.graph_data)):
            point = self.graph_data.get_position(i)
            if point is None:
                continue
            to_point = point - ray_origin
            projection_length = np.dot(to_point, ray_direction)

            if projection_length > 0:
                closest_on_ray = ray_origin + projection_length * ray_direction
                distance_to_ray = np.linalg.norm(point - closest_on_ray)

                if distance_to_ray < closest_distance and distance_to_ray < distance_threshold:
                    closest_distance = distance_to_ray
                    closest_idx = i

        return closest_idx, closest_distance

    def _pick_graph_screen_distance(self, screen_x, screen_y):
        """Fallback graph picking using screen-space distance."""
        if self.graph_data is None or len(self.graph_data) == 0:
            return None, float('inf')

        view_matrix = self.scene.scene.camera.get_view_matrix()
        proj_matrix = self.scene.scene.camera.get_projection_matrix()

        viewport_width = self.scene.frame.width
        viewport_height = self.scene.frame.height

        if viewport_width == 0 or viewport_height == 0:
            return None, float('inf')

        view_mat = np.array(view_matrix).reshape(4, 4)
        proj_mat = np.array(proj_matrix).reshape(4, 4)
        vp_matrix = proj_mat @ view_mat

        closest_idx = None
        closest_screen_dist = float('inf')
        max_screen_dist = 60.0  # Larger threshold for graph vertices

        for i in range(len(self.graph_data)):
            point = self.graph_data.get_position(i)
            if point is None:
                continue
            point_h = np.array([point[0], point[1], point[2], 1.0])
            clip_pos = vp_matrix @ point_h

            if abs(clip_pos[3]) < 1e-6 or clip_pos[3] < 0:
                continue

            ndc = clip_pos[:3] / clip_pos[3]

            if ndc[2] < -1 or ndc[2] > 1:
                continue

            screen_pos_x = (ndc[0] + 1.0) * viewport_width / 2.0
            screen_pos_y = (1.0 - ndc[1]) * viewport_height / 2.0

            dx = screen_pos_x - screen_x
            dy = screen_pos_y - screen_y
            screen_dist = np.sqrt(dx*dx + dy*dy)

            if screen_dist < closest_screen_dist and screen_dist < max_screen_dist:
                closest_screen_dist = screen_dist
                closest_idx = i

        return closest_idx, closest_screen_dist

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.type == gui.KeyEvent.Type.DOWN:
            # Handle Space for nodes, spawns, and graph vertices
            if event.key == gui.KeyName.SPACE:
                if self.selected_graph is not None:
                    self.focus_on_graph(self.selected_graph)
                    return True
                elif self.selected_spawn is not None:
                    self.focus_on_spawn(self.selected_spawn)
                    return True
                elif self.selected_node is not None:
                    self.focus_on_node(self.selected_node)
                    return True
                return False

            # Arrow key navigation only works for nodes
            if self.selected_node is None:
                return False

            links = self.level_data.get_links(self.selected_node)

            link_idx = None

            if event.key == gui.KeyName.UP:
                link_idx = 0
            elif event.key == gui.KeyName.RIGHT:
                link_idx = 1
            elif event.key == gui.KeyName.DOWN:
                link_idx = 2
            elif event.key == gui.KeyName.LEFT:
                link_idx = 3

            if link_idx is not None:
                target_link = links[link_idx]
                if target_link != LevelData.INVALID_LINK and target_link < len(self.level_data):
                    self.inspect_node(target_link, reset_camera=False)
                    return True
                return True

        return False

    def _on_node_selected(self, node_idx):
        """Callback when user selects a node via Go button."""
        self.inspect_node(node_idx, reset_camera=True)

    def _on_coordinate_search(self, x, y, z):
        """Callback for coordinate search."""
        nearest_idx, distance = self.level_data.find_nearest_node(x, y, z)

        if nearest_idx is not None:
            self.inspect_node(nearest_idx, reset_camera=True)
            self.window.close_dialog()
            self.control_panel.set_status(
                f"Found nearest vertex!\nDistance: {distance:.2f} units"
            )
        else:
            DialogFactory.show_error(self.window, "Could not find nearest level vertex")

    def _on_spawn_selected(self, spawn_idx):
        """Callback when user selects a spawn via search."""
        self.inspect_spawn(spawn_idx, move_camera=True)

    def _on_graph_selected(self, graph_idx):
        """Callback when user selects a graph vertex via search."""
        self.inspect_graph(graph_idx, move_camera=True)

    def inspect_node(self, idx, reset_camera=False, move_camera=False):
        """Inspect a node."""
        if 0 <= idx < len(self.level_data):
            # Move camera BEFORE updating selection (so we use old selection as reference)
            if move_camera:
                self.move_camera_to_node(idx)

            self.selected_node = idx

            # Clear spawn selection
            self.selected_spawn = None
            self.control_panel.clear_spawn_selection()
            if self.spawn_data is not None and len(self.spawn_data) > 0:
                self.scene.scene.remove_geometry("spawn_highlight")
                hidden_highlight = self.geometry_manager.hide_spawn_highlight()
                spawn_highlight_mat = rendering.MaterialRecord()
                spawn_highlight_mat.shader = "defaultLitTransparency"
                spawn_highlight_mat.base_color = [1.0, 1.0, 0.0, 0.3]
                self.scene.scene.add_geometry("spawn_highlight", hidden_highlight, spawn_highlight_mat)

                # Hide space_restrictor shapes
                self._update_space_restrictor_shapes(-1)

            # Clear graph selection
            self.selected_graph = None
            self.control_panel.clear_graph_selection()
            if self.graph_data is not None and len(self.graph_data) > 0:
                self.scene.scene.remove_geometry("graph_highlight")
                hidden_highlight = self.geometry_manager.hide_graph_highlight()
                graph_highlight_mat = rendering.MaterialRecord()
                graph_highlight_mat.shader = "defaultLit"
                self.scene.scene.add_geometry("graph_highlight", hidden_highlight, graph_highlight_mat)

            # Reset AI node colors to normal (in case graph was previously selected)
            self.geometry_manager.reset_level_vertex_colors()
            self._update_point_cloud_geometry()

            # Update UI
            self.control_panel.set_current_node(idx)

            # Update highlight
            self.scene.scene.remove_geometry("highlight")
            new_highlight = self.geometry_manager.update_highlight_position(idx)

            highlight_mat = rendering.MaterialRecord()
            highlight_mat.shader = "defaultLit"
            self.scene.scene.add_geometry("highlight", new_highlight, highlight_mat)

            # Update camera
            if reset_camera:
                self.focus_on_node(idx)

            self.scene.force_redraw()

    def inspect_spawn(self, idx, move_camera=False):
        """Inspect a spawn object."""
        if self.spawn_data is None or not (0 <= idx < len(self.spawn_data)):
            return

        # Move camera BEFORE updating selection (so we use old selection as reference)
        if move_camera:
            self.move_camera_to_spawn(idx)

        self.selected_spawn = idx

        # Clear node selection highlight (but keep selected_node for navigation)
        self.scene.scene.remove_geometry("highlight")
        # Move node highlight off-screen
        hidden_node_highlight = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        hidden_node_highlight.paint_uniform_color([1, 0, 0])
        hidden_node_highlight.translate([0, -10000, 0])
        highlight_mat = rendering.MaterialRecord()
        highlight_mat.shader = "defaultLit"
        self.scene.scene.add_geometry("highlight", hidden_node_highlight, highlight_mat)

        # Clear graph selection
        self.selected_graph = None
        self.control_panel.clear_graph_selection()
        if self.graph_data is not None and len(self.graph_data) > 0:
            self.scene.scene.remove_geometry("graph_highlight")
            hidden_graph_highlight = self.geometry_manager.hide_graph_highlight()
            graph_highlight_mat = rendering.MaterialRecord()
            graph_highlight_mat.shader = "defaultLit"
            self.scene.scene.add_geometry("graph_highlight", hidden_graph_highlight, graph_highlight_mat)

        # Reset AI node colors to normal (in case graph was previously selected)
        self.geometry_manager.reset_level_vertex_colors()
        self._update_point_cloud_geometry()

        # Update spawn highlight
        self.scene.scene.remove_geometry("spawn_highlight")
        new_highlight = self.geometry_manager.update_spawn_highlight_position(idx)

        if new_highlight is not None:
            spawn_highlight_mat = rendering.MaterialRecord()
            spawn_highlight_mat.shader = "defaultLitTransparency"
            spawn_highlight_mat.base_color = [1.0, 1.0, 0.0, 0.3]
            self.scene.scene.add_geometry("spawn_highlight", new_highlight, spawn_highlight_mat)

        # Update space_restrictor shapes (show only for selected space_restrictor)
        self._update_space_restrictor_shapes(idx)

        # Update UI
        entity = self.spawn_data.get_entity(idx)
        if entity:
            self.control_panel.set_current_spawn(entity, idx)

        self.scene.force_redraw()

    def _update_space_restrictor_shapes(self, spawn_idx):
        """Show shapes for selected space_restrictor, or hide if not a space_restrictor"""
        # Remove existing geometry if present
        if self.scene.scene.has_geometry("space_restrictor_shapes"):
            self.scene.scene.remove_geometry("space_restrictor_shapes")

        # Get shapes for the selected spawn (returns None if not a space_restrictor/smart_terrain)
        shapes_mesh = self.geometry_manager.get_shapes_for_spawn(spawn_idx)

        if shapes_mesh is not None:
            shape_mat = rendering.MaterialRecord()
            shape_mat.shader = "defaultLitTransparency"
            shape_mat.base_color = [0.0, 1.0, 0.5, 0.3]
            self.scene.scene.add_geometry("space_restrictor_shapes", shapes_mesh, shape_mat)
        else:
            # Add empty mesh to keep the geometry slot
            empty_mesh = self.geometry_manager.hide_space_restrictor_shapes()
            shape_mat = rendering.MaterialRecord()
            shape_mat.shader = "defaultLitTransparency"
            shape_mat.base_color = [0.0, 1.0, 0.5, 0.3]
            self.scene.scene.add_geometry("space_restrictor_shapes", empty_mesh, shape_mat)

    def _update_point_cloud_geometry(self):
        """Update the point cloud geometry in the scene after color changes."""
        self.scene.scene.remove_geometry("point_cloud")
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 5
        self.scene.scene.add_geometry("point_cloud", self.geometry_manager.point_cloud, mat)

    def focus_on_node(self, idx):
        """Focus camera on a node (resets camera perspective)."""
        point = self.level_data.get_point(idx)
        if point is not None:
            extent = 5.0
            min_bound = point - extent
            max_bound = point + extent
            bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            self.scene.setup_camera(60, bounds, point)

    def focus_on_spawn(self, idx):
        """Focus camera on a spawn (resets camera perspective)."""
        if self.spawn_data is None:
            return
        point = self.spawn_data.get_position(idx)
        if point is not None:
            extent = 5.0
            min_bound = point - extent
            max_bound = point + extent
            bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            self.scene.setup_camera(60, bounds, point)

    def inspect_graph(self, idx, move_camera=False):
        """Inspect a game graph vertex."""
        if self.graph_data is None or not (0 <= idx < len(self.graph_data)):
            return

        # Move camera BEFORE updating selection (so we use old selection as reference)
        if move_camera:
            self.move_camera_to_graph(idx)

        self.selected_graph = idx

        # Clear node selection highlight (but keep selected_node for navigation)
        self.scene.scene.remove_geometry("highlight")
        hidden_node_highlight = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        hidden_node_highlight.paint_uniform_color([1, 0, 0])
        hidden_node_highlight.translate([0, -10000, 0])
        highlight_mat = rendering.MaterialRecord()
        highlight_mat.shader = "defaultLit"
        self.scene.scene.add_geometry("highlight", hidden_node_highlight, highlight_mat)

        # Clear spawn selection
        self.selected_spawn = None
        self.control_panel.clear_spawn_selection()
        if self.spawn_data is not None and len(self.spawn_data) > 0:
            self.scene.scene.remove_geometry("spawn_highlight")
            hidden_spawn_highlight = self.geometry_manager.hide_spawn_highlight()
            spawn_highlight_mat = rendering.MaterialRecord()
            spawn_highlight_mat.shader = "defaultLitTransparency"
            spawn_highlight_mat.base_color = [1.0, 1.0, 0.0, 0.3]
            self.scene.scene.add_geometry("spawn_highlight", hidden_spawn_highlight, spawn_highlight_mat)

            # Hide space_restrictor shapes
            self._update_space_restrictor_shapes(-1)

        # Update graph highlight
        self.scene.scene.remove_geometry("graph_highlight")
        new_highlight = self.geometry_manager.update_graph_highlight_position(idx)

        if new_highlight is not None:
            graph_highlight_mat = rendering.MaterialRecord()
            graph_highlight_mat.shader = "defaultLit"
            self.scene.scene.add_geometry("graph_highlight", new_highlight, graph_highlight_mat)

        # Highlight AI nodes belonging to the selected graph vertex
        vertex = self.graph_data.get_vertex(idx)
        target_gvid = vertex.vertex_id if vertex else None

        # Find all level vertices with matching GVID from cross-table
        level_vertex_ids = set()
        if target_gvid is not None and self.level_data.has_cross_table():
            for lvid in range(len(self.level_data)):
                if self.level_data.get_gvid(lvid) == target_gvid:
                    level_vertex_ids.add(lvid)

        self.geometry_manager.highlight_level_vertices(level_vertex_ids)
        self._update_point_cloud_geometry()

        # Update UI
        vertex = self.graph_data.get_vertex(idx)
        if vertex:
            self.control_panel.set_current_graph(vertex, idx)

        self.scene.force_redraw()

    def focus_on_graph(self, idx):
        """Focus camera on a graph vertex (resets camera perspective)."""
        if self.graph_data is None:
            return
        point = self.graph_data.get_position(idx)
        if point is not None:
            extent = 5.0
            min_bound = point - extent
            max_bound = point + extent
            bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            self.scene.setup_camera(60, bounds, point)

    def move_camera_to_graph(self, idx):
        """Move camera to a graph vertex while preserving perspective."""
        if self.graph_data is None:
            return
        new_target = self.graph_data.get_position(idx)
        if new_target is None:
            return
        self._move_camera_to_point(new_target)

    def move_camera_to_node(self, idx):
        """Move camera to a node while preserving perspective (viewing angle and distance)."""
        new_target = self.level_data.get_point(idx)
        if new_target is None:
            return
        self._move_camera_to_point(new_target)

    def move_camera_to_spawn(self, idx):
        """Move camera to a spawn while preserving perspective (viewing angle and distance)."""
        if self.spawn_data is None:
            return
        new_target = self.spawn_data.get_position(idx)
        if new_target is None:
            return
        self._move_camera_to_point(new_target)

    def _move_camera_to_point(self, new_target):
        """Move camera to a point while preserving perspective (viewing angle and distance)."""
        # Get current camera state from view matrix
        view_matrix = np.array(self.scene.scene.camera.get_view_matrix()).reshape(4, 4)

        # Extract camera position (eye) from inverse view matrix
        inv_view = np.linalg.inv(view_matrix)
        current_eye = inv_view[:3, 3]

        # Get current target - prefer graph, then spawn, then node (since node is never cleared)
        if self.selected_graph is not None and self.graph_data is not None:
            current_target = self.graph_data.get_position(self.selected_graph)
        elif self.selected_spawn is not None and self.spawn_data is not None:
            current_target = self.spawn_data.get_position(self.selected_spawn)
        elif self.selected_node is not None:
            current_target = self.level_data.get_point(self.selected_node)
        else:
            # Fallback: estimate target from view direction
            current_target = current_eye - inv_view[:3, 2] * 10  # 10 units in front

        # Calculate offset from target to eye (preserves distance and angle)
        offset = current_eye - current_target

        # Apply same offset to new target
        new_eye = new_target + offset

        # Extract up vector from view matrix
        up = inv_view[:3, 1]

        # Set new camera position
        self.scene.look_at(new_target, new_eye, up)

    def run(self):
        """Run the application."""
        gui.Application.instance.run()
