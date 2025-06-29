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

Spectralign may be obtained by

    pip install spectralign


## Examples

### Two horizontally adjacent tiles

As a most basic example, consider a single slice scanned in two
horizontally adjacent tiles, each of size *W* × *H* pixels, and
with approximately 20% overlap between adjacent tiles. That means that
we should expect an area around the point ***p***₀ = (0.9 *W*, 0.5 *H*)
in the first tile to correspond to a like area around point ***p***₀′
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
regions whos dimensions are an integer power of two. But you could
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

The images aligned above are actually part of a 2 × 3 (rows × columns)
grid of images. For a slightly more involved example, we can align the
whole tableau. First, we load the data. For convenient organization,
we use the (row, column) tuple to index the tiles. (Spectralign will
happily let you use anything that can be a key in a Python dict as a
tile index.)

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
