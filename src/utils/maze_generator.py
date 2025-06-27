import tkinter as tk

class CanvasMazeEditor:
    def __init__(self, master, rows=8, cols=12, cell_size=30):
        self.rows, self.cols, self.cell_size = rows, cols, cell_size
        self.grid = [[0]*cols for _ in range(rows)]

        # Create and place the canvas
        canvas_width = cols * cell_size
        canvas_height = rows * cell_size
        self.canvas = tk.Canvas(master, width=canvas_width, height=canvas_height, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)  # Canvas supports continuous motion events :contentReference[oaicite:0]{index=0}

        # Draw grid rectangles and store their IDs
        self.rects = []
        for r in range(rows):
            row_ids = []
            for c in range(cols):
                x0, y0 = c*cell_size, r*cell_size
                rect_id = self.canvas.create_rectangle(
                    x0, y0, x0+cell_size, y0+cell_size,
                    fill="white", outline="gray"
                )
                row_ids.append(rect_id)
            self.rects.append(row_ids)

        # Bind left-click to toggle, drag to paint only
        self.canvas.bind("<Button-1>", self._on_click)       # toggle on click :contentReference[oaicite:1]{index=1}
        self.canvas.bind("<B1-Motion>", self._on_paint)      # paint-only on drag :contentReference[oaicite:2]{index=2}

        # Export button
        export_btn = tk.Button(master, text="Export 2D Array", command=self.export)
        export_btn.grid(row=1, column=0, sticky="we", padx=10)

    def _on_click(self, event):
        """Toggle cell state on single click."""
        self._toggle_cell(event.x, event.y)

    def _on_paint(self, event):
        """Paint cells black on drag without toggling back to white."""
        self._paint_cell(event.x, event.y)

    def _toggle_cell(self, x, y):
        """Flip between open (0) and wall (1)."""
        c, r = x // self.cell_size, y // self.cell_size
        if 0 <= r < self.rows and 0 <= c < self.cols:
            # Toggle state
            self.grid[r][c] ^= 1
            new_color = "black" if self.grid[r][c] else "white"
            self.canvas.itemconfigure(self.rects[r][c], fill=new_color)

    def _paint_cell(self, x, y):
        """Set cell to wall (1) if it isn’t already."""
        c, r = x // self.cell_size, y // self.cell_size
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.grid[r][c] == 0:
                self.grid[r][c] = 1
                self.canvas.itemconfigure(self.rects[r][c], fill="black")

    def export(self):
        """Print the current 2D array to console."""
        for row in self.grid:
            print(row)
        print("2D array exported.")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Canvas Maze Editor — Paint Only on Drag")
    app = CanvasMazeEditor(root, rows=54, cols=62, cell_size=20)
    root.mainloop()