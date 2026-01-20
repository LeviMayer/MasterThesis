import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

# ── Load your recorded trajectory (including collision flags) from CSV ──
xs, ys, coll_flags = [], [], []
with open("trajectory_with_collisions.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        xs.append(float(row["x"]))
        ys.append(float(row["y"]))
        coll_flags.append(int(row["collision"]))

# ── Count collisions ──
collision_count = sum(coll_flags)

# ── Load the top‐down screenshot ──
img = mpimg.imread("arena_topview.png")
# Falls das Bild vertikal gespiegelt sein sollte, kannst du es umdrehen:
img = img[::-1, :, :]

# ── Specify the world‐coordinate extents that the image covers ──
# (ersetze diese Werte durch die tatsächlichen Boden‐Koordinaten deines Webots‐Szenarios)
xmin, xmax = -12.3,  0.5
ymin, ymax = -13.5,  0

# ── Create the plot ──
fig, ax = plt.subplots(figsize=(8, 8))

# 1) Draw the floor image mit korrektem Extent
ax.imshow(
    img,
    extent=[xmin, xmax, ymin, ymax],  # map image corners → world coords
    origin="lower",                    # damit die untere Bildkante y=ymin entspricht
    zorder=0                           # Hintergrund zuerst zeichnen
)

# 2) Split the trajectory in Kollision / keine Kollision
xs_no_coll = [x for x, c in zip(xs, coll_flags) if c == 0]
ys_no_coll = [y for y, c in zip(ys, coll_flags) if c == 0]

xs_coll = [x for x, c in zip(xs, coll_flags) if c == 1]
ys_coll = [y for y, c in zip(ys, coll_flags) if c == 1]

# 3a) Zeichne Punkte ohne Kollision (z.B. Blau)
ax.scatter(xs_no_coll, ys_no_coll,
           c="blue", s=10, label="Fahrt ohne Kollision", zorder=1)

# 3b) Zeichne Punkte mit Kollision (Rot)
ax.scatter(xs_coll, ys_coll,
           c="red", s=30, label="Kollision", zorder=2)

# 4) Plot‐Formatierung
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("TurtleBot‐Trajektorie mit Kollisions‐Markierung")
ax.set_aspect('equal', 'box')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.7)

# 5) Anzahl der Kollisionen als Text im Plot (oben rechts)
text_x = xmin + 0.02 * (xmax - xmin)    # 2 % vom linken Rand
text_y = ymax - 0.05 * (ymax - ymin)    # 5 % unterhalb des oberen Rands
ax.text(
    text_x, text_y,
    f"Anzahl Kollisionen: {collision_count}",
    color="black",
    fontsize=12,
    bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray", boxstyle="round,pad=0.3")
)

# 6) Legende
ax.legend(loc="upper left")

plt.show()
