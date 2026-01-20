"""
Dialog windows for user interactions.
"""
import open3d.visualization.gui as gui


class DialogFactory:
    """Factory for creating various dialogs."""

    @staticmethod
    def show_error(window, message):
        """Show an error dialog."""
        dialog = gui.Dialog("Error")

        layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        label = gui.Label(message)
        label.text_color = gui.Color(1, 0.5, 0.5)

        ok_button = gui.Button("OK")
        ok_button.set_on_clicked(lambda: window.close_dialog())

        layout.add_child(label)
        layout.add_fixed(20)
        layout.add_child(ok_button)

        dialog.add_child(layout)
        window.show_dialog(dialog)

    @staticmethod
    def show_find_menu(window, on_find_ai_by_vertex_id, on_find_ai_by_xyz,
                       on_find_spawn_by_name, on_find_spawn_by_xyz,
                       on_find_graph_by_gvid, on_find_graph_by_xyz,
                       on_find_patrol_by_name=None, on_find_patrol_by_xyz=None):
        """Show the category selection menu for finding items."""
        dialog = gui.Dialog("Find Something")

        layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        # AI node section
        ai_label = gui.Label("Find AI node:")
        ai_label.text_color = gui.Color(1, 1, 0)
        layout.add_child(ai_label)
        layout.add_fixed(5)

        ai_buttons = gui.Horiz(5)
        ai_vertex_btn = gui.Button("By level vertex ID")
        ai_xyz_btn = gui.Button("By XYZ coordinates")

        def on_ai_vertex():
            window.close_dialog()
            DialogFactory.show_find_ai_by_vertex_id(window, on_find_ai_by_vertex_id)

        def on_ai_xyz():
            window.close_dialog()
            DialogFactory.show_xyz_search(window, "Find AI Node by Coordinates",
                                          on_find_ai_by_xyz)

        ai_vertex_btn.set_on_clicked(on_ai_vertex)
        ai_xyz_btn.set_on_clicked(on_ai_xyz)
        ai_buttons.add_child(ai_vertex_btn)
        ai_buttons.add_child(ai_xyz_btn)
        layout.add_child(ai_buttons)
        layout.add_fixed(15)

        # Spawn section
        spawn_label = gui.Label("Find spawn:")
        spawn_label.text_color = gui.Color(0, 0.8, 0)
        layout.add_child(spawn_label)
        layout.add_fixed(5)

        spawn_buttons = gui.Horiz(5)
        spawn_name_btn = gui.Button("By name")
        spawn_xyz_btn = gui.Button("By XYZ coordinates")

        def on_spawn_name():
            window.close_dialog()
            DialogFactory.show_find_spawn_by_name(window, on_find_spawn_by_name)

        def on_spawn_xyz():
            window.close_dialog()
            DialogFactory.show_xyz_search(window, "Find Spawn by Coordinates",
                                          on_find_spawn_by_xyz)

        spawn_name_btn.set_on_clicked(on_spawn_name)
        spawn_xyz_btn.set_on_clicked(on_spawn_xyz)
        spawn_buttons.add_child(spawn_name_btn)
        spawn_buttons.add_child(spawn_xyz_btn)
        layout.add_child(spawn_buttons)
        layout.add_fixed(15)

        # Graph node section
        graph_label = gui.Label("Find graph node:")
        graph_label.text_color = gui.Color(0.2, 0.5, 1.0)
        layout.add_child(graph_label)
        layout.add_fixed(5)

        graph_buttons = gui.Horiz(5)
        graph_gvid_btn = gui.Button("By game vertex ID")
        graph_xyz_btn = gui.Button("By XYZ coordinates")

        def on_graph_gvid():
            window.close_dialog()
            DialogFactory.show_find_graph_by_gvid(window, on_find_graph_by_gvid)

        def on_graph_xyz():
            window.close_dialog()
            DialogFactory.show_xyz_search(window, "Find Graph Node by Coordinates",
                                          on_find_graph_by_xyz)

        graph_gvid_btn.set_on_clicked(on_graph_gvid)
        graph_xyz_btn.set_on_clicked(on_graph_xyz)
        graph_buttons.add_child(graph_gvid_btn)
        graph_buttons.add_child(graph_xyz_btn)
        layout.add_child(graph_buttons)
        layout.add_fixed(15)

        # Patrol section
        patrol_label = gui.Label("Find patrol:")
        patrol_label.text_color = gui.Color(0.3, 0.3, 0.3)  # Dark grey to match patrol spheres
        layout.add_child(patrol_label)
        layout.add_fixed(5)

        patrol_buttons = gui.Horiz(5)
        patrol_name_btn = gui.Button("By name")
        patrol_xyz_btn = gui.Button("By XYZ coordinates")

        def on_patrol_name():
            window.close_dialog()
            DialogFactory.show_find_patrol_by_name(window, on_find_patrol_by_name)

        def on_patrol_xyz():
            window.close_dialog()
            DialogFactory.show_xyz_search(window, "Find Patrol by Coordinates",
                                          on_find_patrol_by_xyz)

        patrol_name_btn.set_on_clicked(on_patrol_name)
        patrol_xyz_btn.set_on_clicked(on_patrol_xyz)
        patrol_buttons.add_child(patrol_name_btn)
        patrol_buttons.add_child(patrol_xyz_btn)
        layout.add_child(patrol_buttons)
        layout.add_fixed(20)

        # Cancel button
        cancel_button = gui.Button("Cancel")
        cancel_button.set_on_clicked(lambda: window.close_dialog())
        layout.add_child(cancel_button)

        dialog.add_child(layout)
        window.show_dialog(dialog)

    @staticmethod
    def show_find_ai_by_vertex_id(window, on_find_callback):
        """Show dialog to find AI node by vertex ID."""
        dialog = gui.Dialog("Find AI Node by Vertex ID")

        layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        id_label = gui.Label("Level vertex ID:")
        id_input = gui.TextEdit()

        layout.add_child(id_label)
        layout.add_fixed(5)
        layout.add_child(id_input)
        layout.add_fixed(20)

        button_layout = gui.Horiz()
        search_button = gui.Button("Search")
        cancel_button = gui.Button("Cancel")

        def on_search():
            try:
                vertex_id = int(id_input.text_value)
                window.close_dialog()
                on_find_callback(vertex_id)
            except ValueError:
                DialogFactory.show_error(window, "Please enter a valid number")

        search_button.set_on_clicked(on_search)
        cancel_button.set_on_clicked(lambda: window.close_dialog())

        button_layout.add_stretch()
        button_layout.add_child(cancel_button)
        button_layout.add_fixed(10)
        button_layout.add_child(search_button)

        layout.add_child(button_layout)
        dialog.add_child(layout)
        window.show_dialog(dialog)

    @staticmethod
    def show_find_spawn_by_name(window, on_find_callback):
        """Show dialog to find spawn by name."""
        dialog = gui.Dialog("Find Spawn by Name")

        layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        name_label = gui.Label("Spawn name (substring match):")
        name_input = gui.TextEdit()

        layout.add_child(name_label)
        layout.add_fixed(5)
        layout.add_child(name_input)
        layout.add_fixed(20)

        button_layout = gui.Horiz()
        search_button = gui.Button("Search")
        cancel_button = gui.Button("Cancel")

        def on_search():
            name = name_input.text_value.strip()
            if name:
                window.close_dialog()
                on_find_callback(name)
            else:
                DialogFactory.show_error(window, "Please enter a search term")

        search_button.set_on_clicked(on_search)
        cancel_button.set_on_clicked(lambda: window.close_dialog())

        button_layout.add_stretch()
        button_layout.add_child(cancel_button)
        button_layout.add_fixed(10)
        button_layout.add_child(search_button)

        layout.add_child(button_layout)
        dialog.add_child(layout)
        window.show_dialog(dialog)

    @staticmethod
    def show_find_graph_by_gvid(window, on_find_callback):
        """Show dialog to find graph node by game vertex ID."""
        dialog = gui.Dialog("Find Graph Node by Game Vertex ID")

        layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        id_label = gui.Label("Game vertex ID:")
        id_input = gui.TextEdit()

        layout.add_child(id_label)
        layout.add_fixed(5)
        layout.add_child(id_input)
        layout.add_fixed(20)

        button_layout = gui.Horiz()
        search_button = gui.Button("Search")
        cancel_button = gui.Button("Cancel")

        def on_search():
            try:
                gvid = int(id_input.text_value)
                window.close_dialog()
                on_find_callback(gvid)
            except ValueError:
                DialogFactory.show_error(window, "Please enter a valid number")

        search_button.set_on_clicked(on_search)
        cancel_button.set_on_clicked(lambda: window.close_dialog())

        button_layout.add_stretch()
        button_layout.add_child(cancel_button)
        button_layout.add_fixed(10)
        button_layout.add_child(search_button)

        layout.add_child(button_layout)
        dialog.add_child(layout)
        window.show_dialog(dialog)

    @staticmethod
    def show_find_patrol_by_name(window, on_find_callback):
        """Show dialog to find patrol by name."""
        dialog = gui.Dialog("Find Patrol by Name")

        layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        name_label = gui.Label("Patrol name (substring match):")
        name_input = gui.TextEdit()

        layout.add_child(name_label)
        layout.add_fixed(5)
        layout.add_child(name_input)
        layout.add_fixed(20)

        button_layout = gui.Horiz()
        search_button = gui.Button("Search")
        cancel_button = gui.Button("Cancel")

        def on_search():
            name = name_input.text_value.strip()
            if name:
                window.close_dialog()
                on_find_callback(name)
            else:
                DialogFactory.show_error(window, "Please enter a search term")

        search_button.set_on_clicked(on_search)
        cancel_button.set_on_clicked(lambda: window.close_dialog())

        button_layout.add_stretch()
        button_layout.add_child(cancel_button)
        button_layout.add_fixed(10)
        button_layout.add_child(search_button)

        layout.add_child(button_layout)
        dialog.add_child(layout)
        window.show_dialog(dialog)

    @staticmethod
    def show_xyz_search(window, title, on_find_callback):
        """Show dialog to search by XYZ coordinates."""
        dialog = gui.Dialog(title)

        layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        x_label = gui.Label("X coordinate:")
        x_input = gui.TextEdit()
        y_label = gui.Label("Y coordinate:")
        y_input = gui.TextEdit()
        z_label = gui.Label("Z coordinate:")
        z_input = gui.TextEdit()

        layout.add_child(x_label)
        layout.add_fixed(5)
        layout.add_child(x_input)
        layout.add_fixed(10)
        layout.add_child(y_label)
        layout.add_fixed(5)
        layout.add_child(y_input)
        layout.add_fixed(10)
        layout.add_child(z_label)
        layout.add_fixed(5)
        layout.add_child(z_input)
        layout.add_fixed(20)

        button_layout = gui.Horiz()
        find_button = gui.Button("Find Nearest")
        cancel_button = gui.Button("Cancel")

        def on_find():
            try:
                x = float(x_input.text_value)
                y = float(y_input.text_value)
                z = float(z_input.text_value)
                on_find_callback(x, y, z)
            except ValueError:
                DialogFactory.show_error(window, "Please enter valid numbers for X, Y, Z")

        find_button.set_on_clicked(on_find)
        cancel_button.set_on_clicked(lambda: window.close_dialog())

        button_layout.add_stretch()
        button_layout.add_child(cancel_button)
        button_layout.add_fixed(10)
        button_layout.add_child(find_button)

        layout.add_child(button_layout)
        dialog.add_child(layout)
        window.show_dialog(dialog)

    @staticmethod
    def show_spawn_selection(window, matches, on_select_callback):
        """Show dialog to select from multiple spawn matches.

        Args:
            window: The parent window
            matches: List of (index, entity_name) tuples
            on_select_callback: Callback with selected index
        """
        dialog = gui.Dialog("Select Spawn")

        layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        count_label = gui.Label(f"Found {len(matches)} matching spawns:")
        count_label.text_color = gui.Color(0.7, 0.7, 0.7)
        layout.add_child(count_label)
        layout.add_fixed(10)

        # Create scrollable list area
        scroll = gui.ScrollableVert(0, gui.Margins(0, 0, 0, 0))

        for idx, name in matches:
            btn = gui.Button(f"{name} (#{idx})")

            def make_callback(spawn_idx):
                def callback():
                    window.close_dialog()
                    on_select_callback(spawn_idx)
                return callback

            btn.set_on_clicked(make_callback(idx))
            scroll.add_child(btn)
            scroll.add_fixed(2)

        layout.add_child(scroll)
        layout.add_fixed(15)

        cancel_button = gui.Button("Cancel")
        cancel_button.set_on_clicked(lambda: window.close_dialog())
        layout.add_child(cancel_button)

        dialog.add_child(layout)
        window.show_dialog(dialog)

    @staticmethod
    def show_patrol_selection(window, matches, on_select_callback):
        """Show dialog to select from multiple patrol matches.

        Args:
            window: The parent window
            matches: List of (point_index, patrol_name) tuples
            on_select_callback: Callback with selected point index
        """
        dialog = gui.Dialog("Select Patrol")

        layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        count_label = gui.Label(f"Found {len(matches)} matching patrols:")
        count_label.text_color = gui.Color(0.7, 0.7, 0.7)
        layout.add_child(count_label)
        layout.add_fixed(10)

        # Create scrollable list area
        scroll = gui.ScrollableVert(0, gui.Margins(0, 0, 0, 0))

        for idx, name in matches:
            btn = gui.Button(f"{name} (#{idx})")

            def make_callback(point_idx):
                def callback():
                    window.close_dialog()
                    on_select_callback(point_idx)
                return callback

            btn.set_on_clicked(make_callback(idx))
            scroll.add_child(btn)
            scroll.add_fixed(2)

        layout.add_child(scroll)
        layout.add_fixed(15)

        cancel_button = gui.Button("Cancel")
        cancel_button.set_on_clicked(lambda: window.close_dialog())
        layout.add_child(cancel_button)

        dialog.add_child(layout)
        window.show_dialog(dialog)

    @staticmethod
    def show_coordinate_search(window, on_find_callback):
        """Show dialog to search for node by coordinates."""
        dialog = gui.Dialog("Find Level Vertex by Coordinates")

        # Create dialog layout
        layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        # X coordinate
        x_label = gui.Label("X coordinate:")
        x_input = gui.TextEdit()

        # Y coordinate
        y_label = gui.Label("Y coordinate:")
        y_input = gui.TextEdit()

        # Z coordinate
        z_label = gui.Label("Z coordinate:")
        z_input = gui.TextEdit()

        layout.add_child(x_label)
        layout.add_fixed(5)
        layout.add_child(x_input)
        layout.add_fixed(10)
        layout.add_child(y_label)
        layout.add_fixed(5)
        layout.add_child(y_input)
        layout.add_fixed(10)
        layout.add_child(z_label)
        layout.add_fixed(5)
        layout.add_child(z_input)
        layout.add_fixed(20)

        # Buttons
        button_layout = gui.Horiz()
        find_button = gui.Button("Find Nearest")
        cancel_button = gui.Button("Cancel")

        def on_find():
            try:
                x = float(x_input.text_value)
                y = float(y_input.text_value)
                z = float(z_input.text_value)
                on_find_callback(x, y, z)
            except ValueError:
                DialogFactory.show_error(window, "Please enter valid numbers for X, Y, Z")

        def on_cancel():
            window.close_dialog()

        find_button.set_on_clicked(on_find)
        cancel_button.set_on_clicked(on_cancel)

        button_layout.add_stretch()
        button_layout.add_child(cancel_button)
        button_layout.add_fixed(10)
        button_layout.add_child(find_button)

        layout.add_child(button_layout)
        dialog.add_child(layout)

        window.show_dialog(dialog)
