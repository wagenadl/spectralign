# spectralign: Spectral methods for image alignment

Aligning large collections of images is a common task in
microscopy. Whether you have a single slice scanned in several tiles
with a traditional fluorescence microscope or an enormous collection
of images obtained with serial-section electron microscopy, those
images need to be combined into a single 2- or 3-dimensional output.

With spectralign, many such tasks can be accomplished using a single
framework that proceeds according to three steps:

1. **Matching**: Identify matching regions between pairs of
   images. 
2. **Placement**: Optimally place every image in model space.
3. **Rendering**: Blend the source images together to create output.

## Terminology

- **Tile**: A source image. Tiles are commonly organized in z-stacks
  and/or in (row, column) grid layouts. Regardless, we index tiles
  with variable names like *t*, *t*′, etc. Depending on the natural
  structure of your image set, you may consider *t* to be an alias for
  *z* within a stack, or a shorthand for (*r*, *c*) within a slice,
  etc.

- **Z-stack**: A collection of tiles that occupy roughly the same area
  of model space, but at adjacent z levels.

- **Slice**: A collection of tiles that occupy different areas of
  model space at the same z level.

- **Tile set**: All the tiles that make up the image data to be aligned.

- **Point**: An (*x*, *y*) pixel position in a given tile.  Commonly, we
  use variable names like ***p*** and ***p***′ to label points.

- **Tile space**: Coordinates on a tile. Points live in tile space. 

- **Model space**: The 2D canvas onto which tiles will be
  projected. (Even in the case of 3D or 4D tile sets)

- **Location**: A position in model space. 

- **Transformation**: A mapping from coordinates in a tile space to
  the global model space.

With this terminology, we can say that the **Matching** step consists
of identifying pairs of **points** ***p*** and ***p***′ that live in
the **tile spaces** of **tiles** *t* and *t*′ respectively that should
map to the same **location** in **model space**. Meanwhile, the
**Placement** step consists of finding **Transformations** that
realize that goal, and the **Rendering** step produces output images
that represent rectangular areas within **model space**.


## Installation

Spectralign requires Python 3.12 or newer (check with `python --version`).
Then spectralign may be obtained by

    pip install spectralign

Note that on an older Python, pip does not stop with an error: it quietly
installs the old release 0.2.0, which fails with a `SyntaxError` when
imported. If `pip install spectralign` reports version 0.2.0, use a newer
Python (for example in a virtual environment: `python3.12 -m venv .venv`).


## Examples

### Conventions

- Points are `(x, y)` pairs, with `x` to the right and `y` downward. An
  `Image` is an array, so its `shape` is `(height, width)`. Window sizes
  (`size=`) are given as `(width, height)`, or as one integer for a square.
- `spectralign.Image(filename)` reads a file as a grayscale `float32` array
  scaled to 0 to 1 (8-bit and 16-bit files are supported). If you build an
  `Image` from a three-dimensional array instead, the colour channels are
  averaged.
- The tile labels you give to `addmatch` can be any hashable values. The
  positions returned by `Placement.rigid` are the positions of the top left
  corner of each tile in model space, defined up to a common offset (the
  average position is set to zero).
- In releases up to and including 0.4.2, `Matcher.refine` returns three
  values, `p, p_, stats`. The unreleased development version (0.4.3) returns
  `p, p_` unless the `Matcher` was created with `stats=True`. Writing
  `p, p_, *_ = matcher.refine(...)` works with both.

### Trying it without any image files

The next two sections need photographs that are not part of this package. To
check your installation first, this example cuts two overlapping tiles out of
a made-up smooth picture, so that the true offset between them is known:

```python
import numpy as np
from scipy.ndimage import gaussian_filter
import spectralign

rng = np.random.default_rng(1)
scene = gaussian_filter(rng.random((620, 1000)), 3)
scene = ((scene - scene.min()) / (scene.max() - scene.min())).astype(np.float32)

# The second tile starts 400 pixels to the right of and 5 pixels below the first.
img = spectralign.Image(scene[0:600, 0:600])
img_ = spectralign.Image(scene[5:605, 400:1000])

# Start from a rough guess (here: the same place in the overlap) and refine it.
matcher = spectralign.Matcher(img, img_)
p, p_, *_ = matcher.refine((500, 300), (100, 300), size=128, iters=3)

placement = spectralign.Placement()
placement.addmatch(1, 2, p, p_)
pos = placement.rigid()
print(np.array(pos[2]) - np.array(pos[1]))   # approximately [400, 5]
```

### Two horizontally adjacent tiles

As a most basic example, consider a single slice scanned in two
horizontally adjacent tiles, each of size *W* × *H* pixels, and
with approximately 20% overlap between adjacent tiles. That means that
we should expect an area around the point ***p***₀ = (0.9 *W*, 0.5 *H*)
in the first tile to correspond to a like area around point ***p***′₀
= (0.1 *W*, 0.5 *H*) in the second tile. We can inform spectralign about
this situation as follows:

```python
import spectralign
img = spectralign.Image("hedge_11.jpg")
img_ = spectralign.Image("hedge_12.jpg")
H, W = img.shape
p0 = (0.9 * W, 0.5 * H)
p0_ = (0.1 * W, 0.5 * H)
```

(Python variable names may not contain primes (′) so I use an
underscore instead.)

Naturally, these points don′t correspond perfectly. In this particular
case, the images are actually handheld photographs. So we need
spectralign′s matching ability to discover actual matches:

```python
matcher = spectralign.Matcher(img, img_)
p, p_ = matcher.refine(p0, p0_, size=(512, 1024), iters=3)
```

(For technical reasons, spectralign′s matcher is more efficient on
regions whose dimensions are an integer power of two. But you could
have written `size=(0.2 * W, 0.8 * H)` and obtained largely identical
results.)

Now, ***p*** and ***p***′ are spectralign’s estimate of points that
actually match.

If we assume that the only transformation needed to align these images
is a simple linear translation, we may next proceed to place them:

```python
placement = spectralign.Placement()
placement.addmatch(1, 2, p, p_)
pos = placement.rigid()
```

where I chose arbitrary labels *t* = 1 and 2 for the two
tiles. Creating a composite output image is then straightforward:

```python
renderer = spectralign.Renderer(rigid=pos, tilesize=(W, H))
renderer.render(1, img)
renderer.render(2, img_)
out = renderer.image
```

The result is not bad, considering that these images were obtained
with a handheld camera, but we can do better. Since we now know that
***p*** and ***p***′ are closely corresponding points, we can find a
whole grid of matching points, using smaller source rectangles to
obtain more local results:

```python
placement = spectralign.Placement()
for dx in [-50, 50]:
    for dy in range(-H//2 + 100, H//2 - 100, 100):
        p1, p1_ = matcher.refine(p + (dx, dy), p_ + (dx, dy), size=128, iters=3)
        placement.addmatch(1, 2, p1, p1_)
afms = placement.affine()
```

(If you specify `size` as a single integer, the result is a square
area.)

The result is a pair of 
[*affine transformations*](https://en.wikipedia.org/wiki/Affine_transformation)
that map the two tiles onto model space. Rendering output is still
just as easy:

```python
renderer = spectralign.Renderer(affine=afms, tilesize=(W, H))
renderer.render(1, img)
renderer.render(2, img_)
out = renderer.image
```

### A grid of tiles

The images aligned above are actually part of a grid of 2 rows × 3
columns. For a slightly more involved example, we can align the whole
tableau. First, we load the data. For convenient organization, we use
the (row, column) tuple to index the tiles. (Spectralign will happily
let you use anything that can be a key in a Python dict as a tile
index.)

```python
imgs = {}
rows = [1, 2]
cols = [1, 2, 3]
for row in rows:
    for col in cols:
        t = (row, col)
        imgs[t] = spectralign.Image(f"hedge_{row}{col}.jpg")
```

Then we find matching points. The procedure is much the same as
before, except that we now have both horizontal and vertical regions
of overlap between various pairs of tiles.

```python
placement = spectralign.Placement()
H, W = imgs[1, 1].shape

# Horizontally adjacent image pairs
p0 = (0.9 * W, 0.5 * H)
p0_ = (0.1 * W, 0.5 * H)
for row in rows:
    for col in cols[:-1]:
        t = (row, col)
        t_ = (row, col + 1)
        matcher = spectralign.Matcher(imgs[t], imgs[t_])
        p, p_ = matcher.refine(p0, p0_, size=(512, 1024), iters=3)
        placement.addmatch(t, t_, p, p_)

# Vertically adjacent image pairs
p0 = (0.5 * W, 0.9 * H)
p0_ = (0.5 * W, 0.1 * H)
for row in rows[:-1]:
    for col in cols:
        t = (row, col)
        t_ = (row + 1, col)
        matcher = spectralign.Matcher(imgs[t], imgs[t_])
        p, p_ = matcher.refine(p0, p0_, size=(1024, 512), iters=3)
        placement.addmatch(t, t_, p, p_)

# Solve the placement
pos = placement.rigid()
```

Note that we did not bother with tile pairs that only overlap in a
small corner. As an exercise, you could see if considering corner
overlaps improves results.

Finally we render:

```python
renderer = spectralign.Renderer(rigid=pos, tilesize=(W, H))
for row in rows:
    for col in cols:
        t = (row, col)
        renderer.render(t, imgs[t])
out = renderer.image
```

In this case, the cumulative errors from hand-holding the camera
create more visible problems.

Of course we can use the affine transformation approach again:

```python
placement = spectralign.Placement()
H, W = imgs[1, 1].shape

# Horizontally adjacent image pairs
p0 = (0.9 * W, 0.5 * H)
p0_ = (0.1 * W, 0.5 * H)
for row in rows:
    for col in cols[:-1]:
        t = (row, col)
        t_ = (row, col + 1)
        matcher = spectralign.Matcher(imgs[t], imgs[t_])
        p, p_ = matcher.refine(p0, p0_, size=(512, 1024), iters=3)
        for dx in [-50, 50]:
            for dy in range(-H//2 + 100, H//2 - 100, 100):
                p1, p1_ = matcher.refine(p + (dx, dy), p_ + (dx, dy), 128, iters=3)
                placement.addmatch(t, t_, p1, p1_)

# Vertically adjacent image pairs
p0 = (0.5 * W, 0.9 * H)
p0_ = (0.5 * W, 0.1 * H)
for row in rows[:-1]:
    for col in cols:
        t = (row, col)
        t_ = (row + 1, col)
        matcher = spectralign.Matcher(imgs[t], imgs[t_])
        p, p_ = matcher.refine(p0, p0_, size=(1024, 512), iters=3)
        for dx in range(-W//2 + 100, W//2 - 100, 100):
            for dy in [-50, 50]:
                p1, p1_ = matcher.refine(p + (dx, dy), p_ + (dx, dy), 128, iters=3)
                placement.addmatch(t, t_, p1, p1_)

# Solve the placement
afms = placement.affine()
```

and render the result:

```python
renderer = spectralign.Renderer(affine=afms, tilesize=(W, H))
for row in rows:
    for col in cols:
        t = (row, col)
        renderer.render(t, imgs[t])
out = renderer.image
```

### A z-stack of serial sections

For serial-section electron microscopy, the tiles are the sections of a
stack, labelled by their number `z`. Neighbouring sections already overlap
almost completely, so the starting point can simply be the same `(x, y)` in
both sections. This example matches each section with the previous one at a
3 × 3 grid of points, solves for all positions at once, and writes the
aligned sections as PNG files. It assumes that a folder `sections` holds the
images, sorted by name (a public 20-section stack is described below). The
window is 512 pixels: see the note on window size below.

```python
import glob
import os
import spectralign

files = sorted(glob.glob("sections/*.tif"))
imgs = [spectralign.Image(f) for f in files]
H, W = imgs[0].shape

placement = spectralign.Placement()
for z in range(1, len(imgs)):
    matcher = spectralign.Matcher(imgs[z - 1], imgs[z])
    for x in (0.25 * W, 0.5 * W, 0.75 * W):
        for y in (0.25 * H, 0.5 * H, 0.75 * H):
            p, p_, *_ = matcher.refine((x, y), (x, y), size=512, iters=3)
            placement.addmatch(z - 1, z, p, p_)
pos = placement.rigid()

os.makedirs("aligned", exist_ok=True)
renderer = spectralign.Renderer(rigid=pos, tilesize=(W, H))
for z, img in enumerate(imgs):
    renderer.clear()
    renderer.render(z, img)
    renderer.image.save(f"aligned/{z:04d}.png")
```

`renderer.image` is an `Image`, and its `save` method writes a file. For a
stack, call `renderer.clear()` before rendering each section, so that every
section gets its own output picture. The output images all have the same size
and share one coordinate system, so the aligned stack can be opened in any
viewer. Plot `pos` against `z` to see how far the sections had to move.

Choose the window size with care. `refine` returns `stats` as its third value.
If `stats.dist` (the length of the last shift) is more than about a pixel, or
`stats.snr` is far below that of the other matches, the match did not converge,
and a larger window is the first thing to try (see
`help(spectralign.swim.RefinementStats)`). On the public stack below, windows of
128 pixels gave 154 of 171 matches with an `snr` below 8, and positions that
spanned about 44 pixels in x and 46 in y; windows of 512 pixels gave 1 of 171
and spans of about 10 and 14 pixels. To check a result, measure the shift that
is left between neighbouring output sections (for example with
`cv2.phaseCorrelate`): it should be smaller than between the unaligned
sections.

A small stack to try this on: the 20 raw sections of a Drosophila larval
ventral nerve cord (serial-section transmission electron microscopy,
1024 × 1024 pixels) are in `stack1/raw` of
<https://github.com/unidesigner/groundtruth-drosophila-vnc>. Copy the files
into a folder called `sections`. The dataset is licensed CC BY-NC-SA 3.0; its
README allows use for generating or testing non-commercial image segmentation
software, and asks that publications derived from it cite Gerhard, Funke,
Martel, Cardona and Fetter, "Segmented anisotropic ssTEM dataset of neural
tissue" (figshare, 2013, <http://dx.doi.org/10.6084/m9.figshare.856713>).

## Development

Development of spectralign occurs on [github](https://github.com/wagenadl/spectralign).

## Documentation

API documentation for spectralign lives on
[readthedocs](https://spectralign.readthedocs.io/en/latest/).

## Acknowledgments

The algorithm at the core of the “Matching” step is a reimplementation
of the “swim” program in the “SWiFT-IR” package by [Art
Wetzel](https://wp-dev.psc.edu/art-wetzel/) at the Pittsburgh
Supercomputer Center. This method was described in:

> Wetzel AW, Bakal J, Dittrich M, Hildebrand DGC, Morgan HL, Lichtman
> JW. 2016. Registering large volume serial-section electron
> microscopy image sets for neural circuit reconstruction using fft
> signal whitening. Proc. IEEE Applied Imagery Pattern Recognition
> Workshop. [doi: 10.1109/aipr.2016.8010595](https://doi.org/10.1109/AIPR.2016.8010595).
    
That package also contained an antecedent of the `Placement.affine`
method (“mir”), although the approach to handling more than two images
in “SWiFT-IR” is very different from ours.

The core logic of the `Placement.rigid` method derives from a
conversation with [Stephan
Saalfeld](https://www.janelia.org/people/stephan-saalfeld) at
Janelia. This method was described in:

> Ashaber M, Tomina Y, Kassraian P, Bushong EA, Kristan WB, Ellisman
> MH, Wagenaar DA, 2021. Anatomy and activity patterns in a
> multifunctional motor neuron and its surrounding circuits. Elife
> 10, e61881. [doi: 10.7554/eLife.61881](https://doi.org/10.7554/eLife.61881).
