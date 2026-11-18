import math
import os
import sys
import time
import numpy as np
import pygame

# ==================== CONSTANTS & WINDOW SETUP ====================
TITLE = "ASCII Christmas Tree"
CW, CH, FONT_PX = 9, 14, 15
COLS, ROWS = 80, 74
W, H = COLS * CW, ROWS * CH

MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"
HINT = "Space pause · R restart · S screenshot · Esc quit"
DIM, ACCENT, SIGN = (96, 104, 122), (255, 196, 90), (176, 168, 196)
BG = (10, 14, 20)

# Colors (RGB)
WHITE = (250, 252, 255)
SNOW = (220, 235, 250)
GOLD = (255, 215, 65)
SILVER = (225, 235, 245)
RED = (245, 45, 45)
GREEN = (45, 175, 75)
GREEN_DARK = (18, 95, 40)
BROWN = (115, 68, 38)

# Needle glyphs from shadowed (left) to lit (right)
NEEDLE_RAMP = ".,:;!ilYVv^"

T_MSG = 2.4
MESSAGE = "HAPPY NEW YEAR 202(6/7)"

# Screen layout (world units -> character cells)
SX, SY = 28.0, 24.0                # columns / rows per world unit
Y_OFF = 21.0                       # screen row of world y = 0
XZ_FIX = (SY * CH / CW) / SX       # stretch so spheres look round on screen
STAR_Y = 0.64
FLOOR_Y = -1.75
MSG_ROW = ROWS - 6                 # kept clear of the tree and gifts
TREE_TOP, TREE_HEIGHT = 0.55, 1.85  # world y of the tree tip and its height down to the lowest whorl

# Point types
NEEDLE, SNOW_TIP, LIGHT, TINSEL, BAUBLE, BOX, TRUNK, RIBBON, BOW = range(9)
ORNAMENTS = (LIGHT, TINSEL, BAUBLE)
GIFT_FACES = (BOX, RIBBON)

SNOW_RATE = 30.0                   # background snow speeds are given in rows per 1/30 s
GLYPH_CACHE_MAX = 20000            # safety cap; quantized colors keep it far below this


def tree_y(u):
    """World y at fraction u (0 = tip, 1 = lowest whorl) of the tree height."""
    return TREE_TOP - TREE_HEIGHT * u


# ==================== GEOMETRY & SHAPES ====================
def fib_sphere(n):
    """Evenly distributes points on a unit sphere using Fibonacci lattice."""
    pts = []
    phi = math.pi * (math.sqrt(5.0) - 1.0)
    for i in range(n):
        y = 1.0 - (i / float(n - 1 or 1)) * 2.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts.append([math.cos(theta) * radius, y, math.sin(theta) * radius])
    return np.array(pts, dtype=np.float32)


def radial_normals(p, lift=0.4):
    """Outward-facing normals around the tree axis, tilted slightly upward."""
    p = np.atleast_2d(p)
    r = np.hypot(p[:, 0], p[:, 2])
    n = np.column_stack([p[:, 0], lift * r + 0.05, p[:, 2]])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def spread_pick(cands, n, min_dist, rng, avoid=None, weights=None):
    """Randomly picks up to n candidates that are at least min_dist apart.
    Candidates with a higher weight are more likely to be tried first."""
    chosen = []
    taken = np.empty((0, 3), dtype=np.float32) if avoid is None else np.array(avoid)
    if weights is None:
        order = rng.permutation(len(cands))
    else:
        order = np.argsort(-rng.random(len(cands)) ** (1.0 / np.asarray(weights)))
    for k in order:
        p = cands[k]
        if len(taken) == 0 or np.min(np.sum((taken - p) ** 2, axis=1)) >= min_dist ** 2:
            chosen.append(p)
            taken = np.vstack([taken, p])
            if len(chosen) >= n:
                break
    return np.array(chosen)


def box_faces(w, h, d, step):
    """Points and normals on the visible faces (all but bottom) of a box."""
    half = np.array([w, h, d]) / 2.0
    pts, nrm = [], []
    for axis in range(3):
        for sign in (-1, 1):
            if axis == 1 and sign == -1:
                continue
            a, b = [k for k in range(3) if k != axis]
            ga = np.arange(-half[a], half[a] + 1e-6, step)
            gb = np.arange(-half[b], half[b] + 1e-6, step)
            A, B = np.meshgrid(ga, gb)
            p = np.zeros((A.size, 3))
            p[:, a], p[:, b], p[:, axis] = A.ravel(), B.ravel(), sign * half[axis]
            n = np.zeros_like(p)
            n[:, axis] = sign
            pts.append(p)
            nrm.append(n)
    return np.vstack(pts), np.vstack(nrm)


class PointScene:
    def __init__(self):
        self.pts = []
        self.cols = []
        self.normals = []
        self.types = []
        # Surface id (gift box face) a point belongs to, -1 for everything else
        self.surfaces = []

    def add(self, p, col, p_type, norm=None, surface=-1):
        """Adds points; col, p_type and surface may be one value for all points or one per point."""
        p = np.atleast_2d(p)
        n = len(p)
        self.pts.append(p)
        self.cols.append(np.broadcast_to(np.asarray(col, dtype=np.float32), (n, 3)))
        if norm is None:
            self.normals.append(np.zeros((n, 3), dtype=np.float32))
        else:
            self.normals.append(np.broadcast_to(np.asarray(norm, dtype=np.float32), (n, 3)))
        self.types.append(np.broadcast_to(np.asarray(p_type, dtype=np.int32), (n,)))
        self.surfaces.append(np.broadcast_to(np.asarray(surface, dtype=np.int32), (n,)))

    def build(self):
        pts = np.vstack(self.pts).astype(np.float32)
        cols = np.vstack(self.cols).astype(np.float32)
        normals = np.vstack(self.normals).astype(np.float32)
        types = np.concatenate(self.types)
        surfaces = np.concatenate(self.surfaces)
        return pts, cols, normals, types, surfaces


def create_tree():
    rng = np.random.default_rng(7)
    q = PointScene()
    anchors = []

    # 1. Needles: whorls of drooping branches with side twigs.
    # Lower whorls are wider, so they get more branches.
    layers = 22
    golden = math.pi * (3.0 - math.sqrt(5.0))
    k_branch = 0
    for j in range(layers):
        f = j / layers
        v0 = 0.02 + 0.90 * f + rng.uniform(-0.008, 0.008)
        v1 = v0 + 0.055
        nb = int(6 + 20 * f)
        r_layer = 0.95 * (v1 ** 0.85)

        for _ in range(nb):
            tau = k_branch * golden + rng.normal(0, 0.12)
            k_branch += 1
            length = rng.uniform(0.75, 1.0) * r_layer
            n_pts = int(20 + 220 * length)
            s_vals = np.linspace(0.04, 1.0, n_pts)

            droop = 0.22 * (s_vals ** 2)
            px = s_vals * length * math.cos(tau) + rng.normal(0, 0.012, n_pts)
            pz = s_vals * length * math.sin(tau) + rng.normal(0, 0.012, n_pts)
            py = tree_y(v0 + s_vals * (v1 - v0)) - droop
            branch_pts = np.column_stack([px, py, pz])
            tip_mask = s_vals > 0.90
            parts, part_tips = [branch_pts], [tip_mask]

            # Side twigs: longer near the trunk, shorter towards the tip
            for s0 in np.arange(0.22, 0.95, 0.11):
                base = branch_pts[int(s0 * (n_pts - 1))]
                tl = 0.30 * length * (1.05 - s0) + 0.025
                n_tw = max(4, int(tl * 140))
                tt = np.linspace(0.15, 1.0, n_tw)[:, None]
                for twig_side in (-1.0, 1.0):
                    ang = tau + twig_side * rng.uniform(0.55, 0.85)
                    d = np.array([math.cos(ang), -0.30 - 0.6 * s0, math.sin(ang)])
                    parts.append(base + tt * tl * d + rng.normal(0, 0.008, (n_tw, 3)))
                    part_tips.append(tt[:, 0] > 0.8)

            needle_pts = np.vstack(parts)
            n_needles = len(needle_pts)
            snowy = np.concatenate(part_tips) & (rng.random(n_needles) < 0.10)
            needle_cols = np.column_stack([
                rng.integers(15, 40, n_needles),
                rng.uniform(105, 195, n_needles),
                rng.integers(18, 50, n_needles),
            ])
            needle_cols[snowy] = SNOW
            q.add(needle_pts, needle_cols, np.where(snowy, SNOW_TIP, NEEDLE), norm=radial_normals(needle_pts))

            # Outer half of the main branch is where ornaments can hang
            anchors.append(branch_pts[s_vals > 0.70])

    anchors = np.vstack(anchors).astype(np.float32)
    # The wide lower tiers have far more branch tips; favor higher anchors so
    # ornaments spread up the tree instead of piling up near the bottom
    anchor_u = np.clip((TREE_TOP - anchors[:, 1]) / TREE_HEIGHT, 0.0, 1.0)
    anchor_w = (1.12 - anchor_u) ** 3

    # 2. Dense Inner Core
    n_core = 7000
    u_c = rng.uniform(0.02, 0.95, n_core)
    r_c = 0.55 * (u_c ** 0.85) * np.sqrt(rng.uniform(0.05, 1.0, n_core))
    phi_c = rng.uniform(0, 2 * math.pi, n_core)
    core_pts = np.column_stack([r_c * np.cos(phi_c), tree_y(u_c), r_c * np.sin(phi_c)])
    core_cols = np.column_stack([
        rng.integers(10, 30, n_core), rng.integers(65, 125, n_core), rng.integers(15, 38, n_core)
    ])
    q.add(core_pts, core_cols, NEEDLE, norm=radial_normals(core_pts))

    # 3. Tree Trunk (bark surface)
    m_tr = 1200
    y_tr = rng.uniform(FLOOR_Y, -1.20, m_tr)
    phi_tr = rng.uniform(0, 2 * math.pi, m_tr)
    r_tr = 0.10 + rng.normal(0, 0.006, m_tr)
    trunk_pts = np.column_stack([r_tr * np.cos(phi_tr), y_tr, r_tr * np.sin(phi_tr)])
    trunk_cols = np.column_stack([
        rng.integers(85, 120, m_tr), rng.integers(48, 72, m_tr), np.full(m_tr, 28)
    ])
    q.add(trunk_pts, trunk_cols, TRUNK, norm=radial_normals(trunk_pts, lift=0.0))

    # 4. Gold tinsel garland spiralling down the tree, draped in scallops between branches
    turns = 4.2
    u_t = np.linspace(0.06, 0.96, 1600)
    phi_t = turns * 2 * math.pi * u_t
    swag = (0.02 + 0.05 * u_t) * np.abs(np.sin(phi_t * 2.5))
    r_t = 0.94 * (u_t ** 0.85)
    g_pts = np.column_stack([r_t * np.cos(phi_t), tree_y(u_t) - swag, r_t * np.sin(phi_t)])
    q.add(g_pts, GOLD, TINSEL, norm=radial_normals(g_pts))

    # 5. Baubles hanging just in front of the branch tips, one glyph each
    bauble_palettes = [
        (240, 35, 40),   # Crimson Red
        (255, 210, 55),  # Bright Gold
        (220, 230, 248), # Polished Silver
        (60, 125, 245),  # Royal Blue
        (180, 85, 225),  # Violet
    ]
    bauble_anchors = spread_pick(anchors, 80, 0.17, rng, weights=anchor_w)
    for k, a in enumerate(bauble_anchors):
        center = a * [1.06, 1.0, 1.06] - [0, 0.025, 0]
        q.add([center], bauble_palettes[k % len(bauble_palettes)], BAUBLE, norm=radial_normals(center, lift=0.2))

    # 6. Fairy Lights (small glowing points on the outer branches)
    light_palette = [
        (255, 70, 70), (255, 230, 70), (60, 200, 255),
        (255, 130, 220), (255, 255, 255), (90, 255, 140)
    ]
    light_anchors = spread_pick(anchors, 170, 0.09, rng, avoid=bauble_anchors, weights=np.sqrt(anchor_w))
    for i, a in enumerate(light_anchors):
        l_pos = a * [1.04, 1.0, 1.04] + [0, 0.012, 0]
        q.add([l_pos], light_palette[i % len(light_palette)], LIGHT)

    # 7. Gift boxes resting on the floor in a ring around the trunk, listed in ring order.
    # Similar colors (blue/violet, red/violet, gold/silver) sit on opposite sides.
    # (x, z, width, depth, height, box color, ribbon color, yaw)
    gift_defs = [
        (0.53, 0.19, 0.28, 0.24, 0.18, RED, GOLD, 0.3),
        (0.16, 0.49, 0.22, 0.22, 0.19, (40, 95, 190), SILVER, -0.4),
        (-0.32, 0.51, 0.20, 0.18, 0.13, GOLD, RED, 0.9),
        (-0.56, 0.06, 0.26, 0.22, 0.16, GREEN_DARK, RED, -0.2),
        (-0.42, -0.43, 0.18, 0.18, 0.12, (150, 60, 190), GOLD, 0.6),
        (0.08, -0.62, 0.16, 0.16, 0.11, SILVER, (220, 40, 60), 0.0),
        (0.47, -0.30, 0.19, 0.16, 0.15, (20, 160, 165), SILVER, -0.7),
    ]
    bow_sphere = fib_sphere(90)
    for k, (gx, gz, w, d, h, box_col, ribbon_col, yaw) in enumerate(gift_defs):
        box_pts, nrm = box_faces(w, h, d, 0.015)
        is_ribbon = (np.abs(box_pts[:, 0]) < w * 0.09) | (np.abs(box_pts[:, 2]) < d * 0.09)
        surface = k * 6 + np.argmax(np.abs(nrm), axis=1) * 2 + (nrm.sum(axis=1) > 0)

        # Bow: two small loops on top of the lid
        bow = np.vstack([
            bow_sphere * [0.035, 0.022, 0.02] + [loop * 0.03, h / 2 + 0.02, 0]
            for loop in (-1, 1)
        ])
        bow_n = np.vstack([bow_sphere, bow_sphere])

        cy, sy = math.cos(yaw), math.sin(yaw)
        rot = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        offset = np.array([gx, FLOOR_Y + h / 2, gz])
        q.add(box_pts[~is_ribbon] @ rot.T + offset, box_col, BOX, norm=nrm[~is_ribbon] @ rot.T,
              surface=surface[~is_ribbon])
        q.add(box_pts[is_ribbon] @ rot.T + offset, ribbon_col, RIBBON, norm=nrm[is_ribbon] @ rot.T,
              surface=surface[is_ribbon])
        q.add(bow @ rot.T + offset, ribbon_col, BOW, norm=bow_n @ rot.T)

    return q.build()


def depth_priority(types, normals):
    """Small depth bonus that decides which of the points sharing a screen cell is drawn.
    Ornaments win over the needles they hang on, and a gift lid's edge wins the box's
    top row over the side faces, so the top row doesn't flicker while turning."""
    is_lid = np.isin(types, GIFT_FACES) & (np.abs(normals[:, 1]) > 0.5)
    bias = np.zeros(len(types), dtype=np.float32)
    bias[np.isin(types, ORNAMENTS)] = 0.06
    bias[is_lid] = 0.04
    return bias


# ==================== RENDERER & MAIN LOOP ====================
class Scene:
    """Point cloud of the whole scene plus per-point data the renderer needs every frame."""

    def __init__(self, pts, cols, normals, types, surfaces):
        self.pts, self.cols, self.normals = pts, cols, normals
        self.types, self.surfaces = types, surfaces
        self.depth_bias = depth_priority(types, normals)
        # Gift side faces turned away from the camera are hidden by the box itself
        self.gift_side = np.isin(types, GIFT_FACES) & (np.abs(normals[:, 1]) < 0.5)
        self.is_ribbon = types == RIBBON
        self.n_surf = int(surfaces.max()) + 1


def project(scene, angle):
    """Rotates the scene around the tree axis and maps it onto character cells."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot_mat = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32)
    rotated = scene.pts @ rot_mat.T
    rot_normals = scene.normals @ rot_mat.T
    cell_c = np.rint(rotated[:, 0] * SX + COLS / 2.0).astype(np.int64)
    cell_r = np.rint(-rotated[:, 1] * SY + Y_OFF).astype(np.int64)
    return cell_r, cell_c, rotated[:, 2], rot_normals


def visible_points(scene, cell_r, cell_c, depth, rot_normals):
    """Index of the point drawn in every occupied cell (a vectorized z-buffer)."""
    on_screen = np.nonzero(
        (cell_r >= 0) & (cell_r < ROWS) & (cell_c >= 0) & (cell_c < COLS)
        & ~(scene.gift_side & (rot_normals[:, 2] < -0.05))
    )[0]
    cell_key = cell_r * COLS + cell_c
    # One sort by cell, then front to back inside a cell (|depth + bias| < 2 keeps cells apart)
    z = depth[on_screen] + scene.depth_bias[on_screen]
    order = on_screen[np.argsort(cell_key[on_screen] * 8.0 + (2.0 - z.astype(np.float64)), kind="stable")]
    sorted_keys = cell_key[order]
    first = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    visible = order[first]

    # A ribbon is thinner than a cell: wherever a box face shows, use its ribbon
    # instead if the ribbon passes through that cell, so it never blinks out
    rib = on_screen[scene.is_ribbon[on_screen]]
    box_at = np.nonzero(scene.types[visible] == BOX)[0]
    if len(rib) and len(box_at):
        n_surf, surfaces = scene.n_surf, scene.surfaces
        rib_keys, rib_first = np.unique(cell_key[rib] * n_surf + surfaces[rib], return_index=True)
        box_keys = cell_key[visible[box_at]] * n_surf + surfaces[visible[box_at]]
        pos = np.minimum(np.searchsorted(rib_keys, box_keys), len(rib_keys) - 1)
        hit = rib_keys[pos] == box_keys
        visible[box_at[hit]] = rib[rib_first[pos[hit]]]
    return visible


NEEDLE_GLYPHS = np.array(list(NEEDLE_RAMP))
LIGHT_DIR = np.array([0.3, 0.6, 0.74], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)  # toward the camera and slightly above


def shade(scene, visible, t, rot_normals):
    """Glyph and RGB color of every visible point, computed per point type on whole arrays."""
    types = scene.types[visible]
    col = scene.cols[visible]
    dots = np.clip(rot_normals[visible] @ LIGHT_DIR, 0.0, 1.0)
    chars = np.full(len(visible), " ", dtype="<U1")
    rgb = np.zeros((len(visible), 3), dtype=np.float32)

    def part(p_type):
        m = types == p_type
        return m, dots[m], visible[m], col[m]

    # Needles & core: lighting picks the glyph; a per-point jitter breaks up repeated runs
    m, d, idx_pt, c = part(NEEDLE)
    glyph = (d * (len(NEEDLE_RAMP) - 2)).astype(np.int64) + (idx_pt * 7919) % 3 - 1
    chars[m] = NEEDLE_GLYPHS[np.clip(glyph, 0, len(NEEDLE_RAMP) - 1)]
    rgb[m] = c * (0.35 + 0.75 * d)[:, None]

    m, d, idx_pt, c = part(SNOW_TIP)
    chars[m] = "'"
    rgb[m] = c * (0.6 + 0.4 * d)[:, None]

    # Baubles: one round glyph, shaded, with a glint when facing the light
    m, d, idx_pt, c = part(BAUBLE)
    chars[m] = "O"
    rgb[m] = c * (0.7 + 0.6 * d)[:, None] + (120 * d ** 12)[:, None]

    # Fairy lights twinkle
    m, d, idx_pt, c = part(LIGHT)
    blink = np.sin(t * 7.0 + idx_pt * 0.8)
    chars[m] = np.where(blink > 0.2, "*", "+")
    rgb[m] = c * (0.4 + 0.7 * (0.5 + 0.5 * blink))[:, None]

    # Tinsel: a steady gold rope with a soft shimmer
    m, d, idx_pt, c = part(TINSEL)
    chars[m] = "="
    shimmer = 0.85 + 0.15 * np.sin(t * 5.0 + idx_pt * 0.35)
    rgb[m] = c * ((0.65 + 0.5 * d) * shimmer)[:, None]

    # Gift boxes and ribbons: flat-shaded faces, lid edge drawn as "="
    m = np.isin(types, GIFT_FACES)
    d = dots[m]
    is_lid = np.abs(rot_normals[visible[m], 1]) > 0.5
    chars[m] = np.where(is_lid, "=", np.where(d > 0.3, "#", "%"))
    rgb[m] = col[m] * (0.45 + 0.7 * d)[:, None]

    m, d, idx_pt, c = part(BOW)
    chars[m] = "&"
    rgb[m] = c * (0.6 + 0.5 * d)[:, None]

    m, d, idx_pt, c = part(TRUNK)
    chars[m] = np.where(idx_pt % 3 != 0, "|", "!")
    rgb[m] = c * (0.5 + 0.6 * d)[:, None]

    return chars, np.clip(rgb, 0, 255).astype(np.uint8)


def open_window():
    """Opens the window, shrunk to fit the desktop if needed. Returns it and the draw scale."""
    sizes = pygame.display.get_desktop_sizes()
    scale = 1.0
    if sizes and sizes[0][0] > 0 and sizes[0][1] > 0:
        desk_w, desk_h = sizes[0]
        # Leave room for the menu bar, dock / taskbar and the window title bar
        scale = min(1.0, (desk_w - 40) / W, (desk_h - 110) / H)
    size = (max(1, round(W * scale)), max(1, round(H * scale)))
    return pygame.display.set_mode(size), scale


def screenshot_path():
    """A new file name for every screenshot, even several within one second."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path, n = f"christmas_tree_{stamp}.png", 1
    while os.path.exists(path):
        n += 1
        path = f"christmas_tree_{stamp}_{n}.png"
    return path


def main():
    pygame.init()
    window, scale = open_window()
    pygame.display.set_caption(TITLE)
    # Everything is drawn at full size here, then scaled into the window if it had to shrink
    screen = pygame.Surface((W, H)) if scale < 1.0 else window
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(MONO, FONT_PX)
    font_hud = pygame.font.SysFont(MONO, 12)

    scene = Scene(*create_tree())

    # Glyph cache keyed by character and color quantized to 32 levels per channel, so the
    # ever-changing shading, twinkle and shimmer colors reuse a bounded set of glyphs.
    # Glyphs are anti-aliased onto the flat background and blitted with a color key,
    # which is far cheaper than per-pixel alpha blending of thousands of glyphs a frame.
    glyph_cache = {}

    def get_glyph(ch, code):
        key = (ch, code)
        glyph = glyph_cache.get(key)
        if glyph is None:
            if len(glyph_cache) >= GLYPH_CACHE_MAX:
                glyph_cache.clear()
            rgb = ((code >> 10) * 8 + 4, ((code >> 5) & 31) * 8 + 4, (code & 31) * 8 + 4)
            glyph = font.render(ch, True, rgb, BG).convert()
            glyph.set_colorkey(BG)
            glyph_cache[key] = glyph
        return glyph

    # Falling background snow (its own seeded generator, like the tree)
    snow_rng = np.random.default_rng(11)
    n_snow = 120
    snow_x = snow_rng.integers(0, COLS, n_snow)
    snow_y = snow_rng.uniform(0, ROWS, n_snow)
    snow_speed = snow_rng.uniform(0.12, 0.35, n_snow) * SNOW_RATE
    snow_glyphs = np.where(np.arange(n_snow) % 2 == 0, ".", "*")

    # Star sprite on the tree top (rows are taller than columns, so arms are wider)
    star_art = [
        "    |    ",
        "  \\ | /  ",
        "---=*=---",
        "  / | \\  ",
        "    |    ",
    ]
    star_r = int(round(Y_OFF - STAR_Y * SY))
    star_c = COLS // 2
    star_cells = [
        (star_r + r - 2, star_c + c - 4, ch, math.hypot(r - 2, (c - 4) * CW / CH))
        for r, line in enumerate(star_art) for c, ch in enumerate(line) if ch != " "
    ]
    star_cells = [cell for cell in star_cells if 0 <= cell[0] < ROWS and 0 <= cell[1] < COLS]

    start_ticks = pygame.time.get_ticks()
    paused = False
    pause_ticks = 0
    last_t = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    if paused:
                        pause_ticks = pygame.time.get_ticks()
                    else:
                        # Resume where it stopped: shift the start by the time spent paused
                        start_ticks += pygame.time.get_ticks() - pause_ticks
                elif event.key == pygame.K_r:
                    start_ticks = pause_ticks = pygame.time.get_ticks()
                    last_t = 0.0
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, screenshot_path())

        if paused:
            clock.tick(30)
            continue

        t = (pygame.time.get_ticks() - start_ticks) / 1000.0
        dt = min(max(t - last_t, 0.0), 0.25)
        last_t = t
        screen.fill(BG)

        grid_char = np.full((ROWS, COLS), " ", dtype="<U1")
        grid_rgb = np.full((ROWS, COLS, 3), WHITE, dtype=np.uint8)

        # --- 1. Background snowfall (moves by elapsed time, not per frame) ---
        snow_y = (snow_y + snow_speed * dt) % ROWS
        snow_r = snow_y.astype(np.int64)
        keep = np.abs(snow_r - MSG_ROW) > 1  # keep the greeting readable
        grid_char[snow_r[keep], snow_x[keep]] = snow_glyphs[keep]
        grid_rgb[snow_r[keep], snow_x[keep]] = (150, 170, 195)

        # --- 2. Tree, ornaments and gifts ---
        angle = t * 0.30
        cell_r, cell_c, depth, rot_normals = project(scene, angle)
        visible = visible_points(scene, cell_r, cell_c, depth, rot_normals)
        chars, rgb = shade(scene, visible, t, rot_normals)
        grid_char[cell_r[visible], cell_c[visible]] = chars
        grid_rgb[cell_r[visible], cell_c[visible]] = rgb
        active_letters = len(visible)

        # --- 3. Star on the tree top, shown from the first frame with a soft twinkle ---
        for r, c, ch, dist in star_cells:
            power = 1.0 - 0.16 * dist + 0.18 * math.sin(t * 3.0 - dist * 1.5)
            base = (255, 250, 225) if dist == 0 else (255, 212, 70)
            grid_char[r, c] = ch
            grid_rgb[r, c] = [min(255, max(0, int(v * power))) for v in base]

        # --- 4. "HAPPY NEW YEAR" Typewriter Message ---
        if t > T_MSG:
            shown = min(len(MESSAGE), int((t - T_MSG) * 9))
            x0 = (COLS - len(MESSAGE)) // 2

            for i, ch in enumerate(MESSAGE[:shown]):
                if ch != " ":
                    grid_char[MSG_ROW, x0 + i] = ch
                    grid_rgb[MSG_ROW, x0 + i] = [RED, GOLD, (60, 230, 90)][(i // 2 + int(t * 1.5)) % 3]

        # --- 5. Render cell grid in one batched blit ---
        rows, cols = np.nonzero(grid_char != " ")
        q = (grid_rgb[rows, cols] >> 3).astype(np.int64)
        codes = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
        screen.blits(
            [
                (get_glyph(ch, code), (c * CW, r * CH))
                for ch, code, r, c in zip(
                    grid_char[rows, cols].tolist(), codes.tolist(), rows.tolist(), cols.tolist()
                )
            ],
            doreturn=False,
        )

        # --- 6. Authentic Video HUD & Watermark ---
        rot_deg = (math.degrees(angle)) % 360.0
        hud_rot = font_hud.render(f"rotation {rot_deg:06.2f}°", True, DIM)
        hud_pts = font_hud.render(f"letters {active_letters:,}", True, DIM)
        hud_hint = font_hud.render(HINT, True, DIM)

        screen.blit(hud_rot, (14, 12))
        screen.blit(hud_pts, (14, 28))
        screen.blit(hud_hint, (W // 2 - hud_hint.get_width() // 2, H - 22))

        if screen is not window:
            pygame.transform.smoothscale(screen, window.get_size(), window)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
