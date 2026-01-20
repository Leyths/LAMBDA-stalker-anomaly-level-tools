"""
Control panel UI for node inspection (read-only).
Rebuilds panel content when switching between node, spawn, and graph selection modes.
"""
from typing import Optional, Callable
import open3d.visualization.gui as gui

from core.data_loader import LevelData, SpawnData, GraphData, PatrolData
from utils.helpers import format_node_info, format_spawn_info, format_graph_vertex_info, format_patrol_point_info
from .dialogs import DialogFactory


class ControlPanel:
    """Right-side control panel for node inspection."""

    # Selection modes
    MODE_NONE = 0
    MODE_NODE = 1
    MODE_SPAWN = 2
    MODE_GRAPH = 3
    MODE_PATROL = 4

    def __init__(self, level_data: LevelData, on_node_selected, on_coordinate_search,
                 spawn_data: Optional[SpawnData] = None,
                 graph_data: Optional[GraphData] = None,
                 patrol_data: Optional[PatrolData] = None,
                 on_panel_rebuild: Optional[Callable] = None,
                 on_spawn_selected: Optional[Callable] = None,
                 on_graph_selected: Optional[Callable] = None,
                 on_patrol_selected: Optional[Callable] = None):
        self.level_data = level_data
        self.spawn_data = spawn_data
        self.graph_data = graph_data
        self.patrol_data = patrol_data
        self.on_node_selected = on_node_selected
        self.on_coordinate_search = on_coordinate_search
        self.on_panel_rebuild = on_panel_rebuild  # Callback when panel needs to be swapped
        self.on_spawn_selected = on_spawn_selected
        self.on_graph_selected = on_graph_selected
        self.on_patrol_selected = on_patrol_selected

        self.current_node_idx = None
        self.current_spawn_idx = None
        self.current_graph_idx = None
        self.current_patrol_idx = None
        self.current_patrol_name = None
        self.current_patrol_connected = []
        self.selection_mode = self.MODE_NONE
        self.window = None

        # Build initial panel
        self.panel = None
        self._build_panel()

    def _build_panel(self):
        """Build the panel for the current mode."""
        self.panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        # Title
        title_label = gui.Label("Level Inspector")
        title_label.text_color = gui.Color(1, 1, 0)
        self.panel.add_child(title_label)
        self.panel.add_fixed(10)

        # Instructions - vary based on mode
        if self.selection_mode in (self.MODE_SPAWN, self.MODE_GRAPH, self.MODE_PATROL):
            instruction_text = (
                "Controls:\n"
                "- Left Click+Drag: Rotate camera\n"
                "- Right Click+Drag: Pan camera\n"
                "- Ctrl+Click: Select item\n"
                "- Space: Focus on selection\n"
            )
        else:
            instruction_text = (
                "Controls:\n"
                "- Left Click+Drag: Rotate camera\n"
                "- Right Click+Drag: Pan camera\n"
                "- Ctrl+Click: Select item\n"
                "- Arrow Keys: Navigate vertex links\n"
                "- Space: Focus on selection\n"
            )
        instruction_label = gui.Label(instruction_text)
        instruction_label.text_color = gui.Color(0.7, 0.7, 0.7)
        self.panel.add_child(instruction_label)
        self.panel.add_fixed(10)

        # Node navigation controls (always visible)
        self._add_node_controls()

        # Mode-specific content
        if self.selection_mode == self.MODE_SPAWN:
            self._add_spawn_content()
        elif self.selection_mode == self.MODE_GRAPH:
            self._add_graph_content()
        elif self.selection_mode == self.MODE_PATROL:
            self._add_patrol_content()
        else:
            self._add_node_content()

    def _add_node_controls(self):
        """Add node navigation controls (always present)."""
        find_button = gui.Button("Find something")
        find_button.set_on_clicked(self._on_find_clicked)
        self.panel.add_child(find_button)
        self.panel.add_fixed(15)

    def _add_node_content(self):
        """Add node-specific content."""
        # Selection type
        if self.selection_mode == self.MODE_NODE:
            type_label = gui.Label("Level Vertex")
            type_label.text_color = gui.Color(1, 1, 0)
        else:
            type_label = gui.Label("Nothing selected")
            type_label.text_color = gui.Color(0.7, 0.7, 0.7)
        self.panel.add_child(type_label)
        self.panel.add_fixed(10)

        # Info display
        if self.selection_mode == self.MODE_NODE and self.current_node_idx is not None:
            point = self.level_data.get_point(self.current_node_idx)
            cover_score = self.level_data.get_color_score(self.current_node_idx)
            gvid = self.level_data.get_gvid(self.current_node_idx)
            info_text = format_node_info(self.current_node_idx, point, cover_score, gvid)
        else:
            info_text = self._get_default_info_text()

        self.info_label = gui.Label(info_text)
        self.info_label.text_color = gui.Color(1, 1, 1)
        self.panel.add_child(self.info_label)
        self.panel.add_fixed(15)

        # Links section
        links_label = gui.Label("Links:")
        links_label.text_color = gui.Color(0.8, 0.8, 0.8)
        self.panel.add_child(links_label)
        self.panel.add_fixed(5)

        self.link_directions = ["Up", "Right", "Down", "Left"]
        self.link_buttons = []

        # Get current links if we have a selected node
        links = None
        if self.current_node_idx is not None:
            links = self.level_data.get_links(self.current_node_idx)

        for i, direction in enumerate(self.link_directions):
            layout = gui.Horiz()
            dir_label = gui.Label(f"{direction}:")
            dir_label.text_color = gui.Color(0.7, 0.7, 0.7)

            # Determine button text and state
            if links is not None:
                link_id = links[i]
                if link_id == LevelData.INVALID_LINK:
                    btn_text = "None"
                    btn_enabled = False
                else:
                    btn_text = str(link_id)
                    btn_enabled = True
            else:
                btn_text = "None"
                btn_enabled = False

            link_button = gui.Button(btn_text)
            link_button.enabled = btn_enabled
            link_button.set_on_clicked(self._make_link_navigate_callback(i))
            self.link_buttons.append(link_button)

            layout.add_child(dir_label)
            layout.add_stretch()
            layout.add_child(link_button)

            self.panel.add_child(layout)
            self.panel.add_fixed(3)

    def _add_spawn_content(self):
        """Add spawn-specific content."""
        # Selection type
        type_label = gui.Label("Spawn Object")
        type_label.text_color = gui.Color(0, 0.8, 0)
        self.panel.add_child(type_label)
        self.panel.add_fixed(10)

        # Spawn count
        if self.spawn_data is not None:
            count_label = gui.Label(f"{len(self.spawn_data)} spawns on this level")
            count_label.text_color = gui.Color(0.7, 0.7, 0.7)
            self.panel.add_child(count_label)
            self.panel.add_fixed(10)

        # Spawn info
        if self.current_spawn_idx is not None and self.spawn_data is not None:
            entity = self.spawn_data.get_entity(self.current_spawn_idx)
            if entity:
                info_text = format_spawn_info(entity)
            else:
                info_text = "Error loading spawn data"
        else:
            info_text = "Click a spawn point to see details"

        self.info_label = gui.Label(info_text)
        self.info_label.text_color = gui.Color(1, 1, 1)
        self.panel.add_child(self.info_label)

    def _add_graph_content(self):
        """Add game graph vertex-specific content."""
        # Selection type
        type_label = gui.Label("Game Graph Vertex")
        type_label.text_color = gui.Color(0.2, 0.5, 1.0)  # Bright blue
        self.panel.add_child(type_label)
        self.panel.add_fixed(10)

        # Graph vertex count
        if self.graph_data is not None:
            count_label = gui.Label(f"{len(self.graph_data)} graph vertices on this level")
            count_label.text_color = gui.Color(0.7, 0.7, 0.7)
            self.panel.add_child(count_label)
            self.panel.add_fixed(10)

        # Graph vertex info
        if self.current_graph_idx is not None and self.graph_data is not None:
            vertex = self.graph_data.get_vertex(self.current_graph_idx)
            if vertex:
                edges_info = self.graph_data.get_edges_info(self.current_graph_idx)
                info_text = format_graph_vertex_info(vertex, edges_info, self.current_graph_idx)
            else:
                info_text = "Error loading graph vertex data"
        else:
            info_text = "Click a graph vertex to see details"

        self.info_label = gui.Label(info_text)
        self.info_label.text_color = gui.Color(1, 1, 1)
        self.panel.add_child(self.info_label)

    def _get_default_info_text(self):
        """Get default info text based on available data."""
        spawn_info = ""
        if self.spawn_data is not None and len(self.spawn_data) > 0:
            spawn_info = f"\n\n{len(self.spawn_data)} spawn objects on this level"
        return f"Ctrl+Click to select a vertex or spawn{spawn_info}"

    def _request_rebuild(self):
        """Request the main window to swap in the new panel."""
        old_panel = self.panel
        self._build_panel()
        if self.on_panel_rebuild:
            self.on_panel_rebuild(old_panel, self.panel)

    def _on_find_clicked(self):
        """Handle find something button click - shows category menu."""
        if self.window:
            DialogFactory.show_find_menu(
                self.window,
                on_find_ai_by_vertex_id=self._on_find_ai_by_vertex_id,
                on_find_ai_by_xyz=self._on_find_ai_by_xyz,
                on_find_spawn_by_name=self._on_find_spawn_by_name,
                on_find_spawn_by_xyz=self._on_find_spawn_by_xyz,
                on_find_graph_by_gvid=self._on_find_graph_by_gvid,
                on_find_graph_by_xyz=self._on_find_graph_by_xyz,
                on_find_patrol_by_name=self._on_find_patrol_by_name,
                on_find_patrol_by_xyz=self._on_find_patrol_by_xyz
            )

    def _on_find_ai_by_vertex_id(self, vertex_id):
        """Handle find AI node by vertex ID."""
        self.on_node_selected(vertex_id)

    def _on_find_ai_by_xyz(self, x, y, z):
        """Handle find AI node by XYZ coordinates."""
        self.on_coordinate_search(x, y, z)

    def _on_find_spawn_by_name(self, name):
        """Handle find spawn by name."""
        if self.spawn_data is None:
            return

        matches = self.spawn_data.find_by_name(name)
        if not matches:
            DialogFactory.show_error(self.window, f"No spawns found matching '{name}'")
            return

        if len(matches) == 1:
            # Single match - go directly
            idx, _ = matches[0]
            if self.on_spawn_selected:
                self.on_spawn_selected(idx)
        else:
            # Multiple matches - show selection dialog
            DialogFactory.show_spawn_selection(
                self.window,
                matches,
                lambda idx: self.on_spawn_selected(idx) if self.on_spawn_selected else None
            )

    def _on_find_spawn_by_xyz(self, x, y, z):
        """Handle find spawn by XYZ coordinates."""
        if self.spawn_data is None:
            DialogFactory.show_error(self.window, "No spawn data loaded")
            return

        nearest_idx, distance = self.spawn_data.find_nearest_spawn(x, y, z)
        if nearest_idx is not None:
            if self.on_spawn_selected:
                self.on_spawn_selected(nearest_idx)
            self.window.close_dialog()
            self.set_status(f"Found nearest spawn!\nDistance: {distance:.2f} units")
        else:
            DialogFactory.show_error(self.window, "Could not find nearest spawn")

    def _on_find_graph_by_gvid(self, gvid):
        """Handle find graph node by game vertex ID."""
        if self.graph_data is None:
            DialogFactory.show_error(self.window, "No graph data loaded")
            return

        # Find the local index for this global vertex ID
        for idx in range(len(self.graph_data)):
            vertex = self.graph_data.get_vertex(idx)
            if vertex and vertex.vertex_id == gvid:
                if self.on_graph_selected:
                    self.on_graph_selected(idx)
                return

        DialogFactory.show_error(self.window, f"Game vertex ID {gvid} not found on this level")

    def _on_find_graph_by_xyz(self, x, y, z):
        """Handle find graph node by XYZ coordinates."""
        if self.graph_data is None:
            DialogFactory.show_error(self.window, "No graph data loaded")
            return

        nearest_idx, distance = self.graph_data.find_nearest_vertex(x, y, z)
        if nearest_idx is not None:
            if self.on_graph_selected:
                self.on_graph_selected(nearest_idx)
            self.window.close_dialog()
            self.set_status(f"Found nearest graph vertex!\nDistance: {distance:.2f} units")
        else:
            DialogFactory.show_error(self.window, "Could not find nearest graph vertex")

    def _on_find_patrol_by_name(self, name):
        """Handle find patrol by name."""
        if self.patrol_data is None:
            return

        matches = self.patrol_data.find_by_name(name)
        if not matches:
            DialogFactory.show_error(self.window, f"No patrols found matching '{name}'")
            return

        if len(matches) == 1:
            # Single match - go directly
            idx, _ = matches[0]
            if self.on_patrol_selected:
                self.on_patrol_selected(idx)
        else:
            # Multiple matches - show selection dialog
            DialogFactory.show_patrol_selection(
                self.window,
                matches,
                lambda idx: self.on_patrol_selected(idx) if self.on_patrol_selected else None
            )

    def _on_find_patrol_by_xyz(self, x, y, z):
        """Handle find patrol by XYZ coordinates."""
        if self.patrol_data is None:
            DialogFactory.show_error(self.window, "No patrol data loaded")
            return

        nearest_idx, distance = self.patrol_data.find_nearest_point(x, y, z)
        if nearest_idx is not None:
            if self.on_patrol_selected:
                self.on_patrol_selected(nearest_idx)
            self.window.close_dialog()
            self.set_status(f"Found nearest patrol point!\nDistance: {distance:.2f} units")
        else:
            DialogFactory.show_error(self.window, "Could not find nearest patrol point")

    def _make_link_navigate_callback(self, link_index):
        """Create callback for navigating to a linked node."""
        def callback():
            if self.current_node_idx is None:
                return

            links = self.level_data.get_links(self.current_node_idx)
            target_link = links[link_index]

            if target_link != LevelData.INVALID_LINK and target_link < len(self.level_data):
                self.on_node_selected(target_link)

        return callback

    def set_current_node(self, idx):
        """Update the UI for the current node."""
        old_mode = self.selection_mode
        self.current_node_idx = idx
        self.current_spawn_idx = None
        self.current_graph_idx = None
        self.selection_mode = self.MODE_NODE

        # Rebuild panel if mode changed
        if old_mode != self.MODE_NODE:
            self._request_rebuild()
        else:
            # Just update the content
            point = self.level_data.get_point(idx)
            cover_score = self.level_data.get_color_score(idx)
            gvid = self.level_data.get_gvid(idx)
            self.info_label.text = format_node_info(idx, point, cover_score, gvid)
            self._update_links(idx)

    def _update_links(self, idx):
        """Update the link buttons for a node."""
        links = self.level_data.get_links(idx)
        if links is None or not hasattr(self, 'link_buttons'):
            return

        for i in range(4):
            link_id = links[i]
            if link_id == LevelData.INVALID_LINK:
                self.link_buttons[i].text = "None"
                self.link_buttons[i].enabled = False
            else:
                self.link_buttons[i].text = str(link_id)
                self.link_buttons[i].enabled = True

    def set_status(self, message):
        """Set a status message in the info area."""
        if hasattr(self, 'info_label'):
            current_info = self.info_label.text
            self.info_label.text = f"{message}\n\n{current_info}"

    def set_frame(self, x, y, width, height):
        """Set the panel frame."""
        self.panel.frame = gui.Rect(x, y, width, height)

    def set_window(self, window):
        """Set the window reference for dialogs."""
        self.window = window

    def set_current_spawn(self, entity, idx=None):
        """Update the UI for a selected spawn object."""
        old_mode = self.selection_mode
        self.current_spawn_idx = idx
        self.current_node_idx = None
        self.current_graph_idx = None
        self.selection_mode = self.MODE_SPAWN

        # Rebuild panel if mode changed
        if old_mode != self.MODE_SPAWN:
            self._request_rebuild()
        else:
            # Just update the content
            info_text = format_spawn_info(entity)
            self.info_label.text = info_text

    def clear_spawn_selection(self):
        """Clear the spawn selection display."""
        self.current_spawn_idx = None
        if self.selection_mode == self.MODE_SPAWN:
            self.selection_mode = self.MODE_NONE
            self._request_rebuild()

    def set_current_graph(self, vertex, idx=None):
        """Update the UI for a selected game graph vertex."""
        old_mode = self.selection_mode
        self.current_graph_idx = idx
        self.current_node_idx = None
        self.current_spawn_idx = None
        self.selection_mode = self.MODE_GRAPH

        # Rebuild panel if mode changed
        if old_mode != self.MODE_GRAPH:
            self._request_rebuild()
        else:
            # Just update the content
            if self.graph_data is not None and idx is not None:
                edges_info = self.graph_data.get_edges_info(idx)
                info_text = format_graph_vertex_info(vertex, edges_info, idx)
                self.info_label.text = info_text

    def clear_graph_selection(self):
        """Clear the graph selection display."""
        self.current_graph_idx = None
        if self.selection_mode == self.MODE_GRAPH:
            self.selection_mode = self.MODE_NONE
            self._request_rebuild()

    def _add_patrol_content(self):
        """Add patrol point-specific content."""
        # Selection type
        type_label = gui.Label("Patrol Path Point")
        type_label.text_color = gui.Color(0.3, 0.3, 0.3)  # Dark grey to match patrol spheres
        self.panel.add_child(type_label)
        self.panel.add_fixed(10)

        # Patrol point count
        if self.patrol_data is not None:
            count_label = gui.Label(f"{len(self.patrol_data)} patrol points on this level")
            count_label.text_color = gui.Color(0.7, 0.7, 0.7)
            self.panel.add_child(count_label)
            self.panel.add_fixed(10)

        # Patrol point info
        if self.current_patrol_idx is not None and self.patrol_data is not None:
            point = self.patrol_data.get_point(self.current_patrol_idx)
            if point:
                info_text = format_patrol_point_info(
                    point,
                    self.current_patrol_name,
                    self.current_patrol_idx,
                    self.current_patrol_connected
                )
            else:
                info_text = "Error loading patrol point data"
        else:
            info_text = "Click a patrol point to see details"

        self.info_label = gui.Label(info_text)
        self.info_label.text_color = gui.Color(1, 1, 1)
        self.panel.add_child(self.info_label)

    def set_current_patrol(self, point, patrol_name: str, idx: int, connected_points: list):
        """Update the UI for a selected patrol point."""
        old_mode = self.selection_mode
        self.current_patrol_idx = idx
        self.current_patrol_name = patrol_name
        self.current_patrol_connected = connected_points
        self.current_node_idx = None
        self.current_spawn_idx = None
        self.current_graph_idx = None
        self.selection_mode = self.MODE_PATROL

        # Rebuild panel if mode changed
        if old_mode != self.MODE_PATROL:
            self._request_rebuild()
        else:
            # Just update the content
            info_text = format_patrol_point_info(point, patrol_name, idx, connected_points)
            self.info_label.text = info_text

    def clear_patrol_selection(self):
        """Clear the patrol selection display."""
        self.current_patrol_idx = None
        self.current_patrol_name = None
        self.current_patrol_connected = []
        if self.selection_mode == self.MODE_PATROL:
            self.selection_mode = self.MODE_NONE
            self._request_rebuild()
