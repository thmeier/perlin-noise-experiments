import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from typing import Tuple, List

# TODO: make operations memory-efficient, i.e. compute stuff in-place

def normalize(X: np.ndarray, axis: int) -> np.ndarray:
    magnitudes = np.expand_dims(
        np.sqrt(np.sum(np.square(X), axis=axis))
    , axis=axis)
    return np.divide(X, magnitudes)

#def generate_noise(grid: Tuple[int], resolution: Tuple[int]) -> np.ndarray:
if __name__ == '__main__':
    #grid, resolution = (32,), (1000,)
    #grid, resolution = (16,16), (1000,1000)
    grid, resolution = (6,6,6), (100,101,102)

    # random number generator for debugging
    rng = np.random.default_rng(seed=42)

    n, n_res = len(grid), len(resolution)

    # TODO: allow n_res == 1, to set resolution for all dimensions
    assert n == n_res, f"missing resolution per dimension"
    assert n > 0, f"empty grid silly goose"

    # has shape (n_grid,) + resolution
    if n > 1:
        mesh = np.array(np.meshgrid(*[
            np.linspace(0, grid_end-1, res)
            for grid_end, res in zip(grid, resolution)
        # NOTE:
        # swapaxes is needed as traditionally, the x-axis and the y-axis are swapped
        # by meshgrid() since numpy has the convention that the y-axis is indexed 
        # first, then only the x-axis.
        # Since the very first dimension however is the n_grid dimension, the 
        # x-axis and y-axis are axis 2 and 1 before swapping.
        ])).swapaxes(1, 2)
    else:
        mesh = np.linspace(0, grid[0] - 1, resolution[0]).reshape(1, -1)

    # offset vectors from grid cells to mesh point

    floors = np.floor(mesh)

    # corner coordinates of a n_grid dimensional hyper cube
    # i.e. [(0,0), (0,1), (1,0), (1,1)] for 2D case
    # there are 2 ** n_grid many corners
    neighbours = np.expand_dims(np.array(list(
        it.product(*[ [0,1] for _ in range(n) ])
    )).T, axis=tuple(np.arange(n)+2))

    # for every mesh point store the 2**n enclosing grid points
    mesh_neighbours = np.expand_dims(floors, axis=1) + neighbours

    # for every mesh point comute the offset vectors to each of the 2**n_grid enclosing grid points
    offsets = np.expand_dims(mesh, axis=1) - mesh_neighbours

    # this ensures that the enclosing grid points of the rightmost mesh points 
    # wrap around to the leftmost grid points
    # otherwise we get out of bounds errors
    mesh_neighbours_idx = np.mod(
        mesh_neighbours,
        # TODO: beautify
        np.array(grid).reshape(-1, *((1,) * (n+1)))
    ).astype(int)

    gradients_grid = rng.random((n,) + grid) * 2 - 1
    if n > 1:
        # make each gradient an unit-length vector
        gradients_grid = normalize(gradients_grid, axis=0)

    gradients_mesh = gradients_grid[:, *mesh_neighbours_idx]

    # dot product between each gradient and offset vector per corner per point in mesh
    # has shape (2 ** n_grid, *resolution)
    dot_products = np.einsum('nc...,nc...->c...', gradients_mesh, offsets)

    # interpolation

    # this is a linear interpolation, but with smoothsteped mesh points
    # mesh - floors assures that we only have the fractional part of each mesh point,
    # since perlin's fade fucntion assumes inputs are in the range [0,1]
    # interpolation using perlin's fade function: 6x^5 - 15x^4 + 10x^3
    fade = lambda x: x * x * x * (x * (x * 6 - 15) + 10)
    smoothstep = lambda x: x * x * (3 - 2 * x)
    linear = lambda x: x
    #smooth_mesh_fractions = smoothstep(mesh - floors)
    smooth_mesh_fractions = fade(mesh - floors)

    # interpolate one dimension after another
    for i, dim in enumerate(range(n, 0, -1)):

        # 2 ** (dim - 1) to select the face of the dim-dimensional hypercube along the first dimension
        _num_corners_hyperface = 2 ** (dim - 1)
        # enumeration of the corners of the dim-dimensional hypercube
        _corner_idx = np.arange(2 ** dim)
        # via the binary encoding of the  hypercube's corners allows us to select
        # all the relevant corners of the current face
        # intuitively, a binary encoding of a dim-dimensional hypercube corner
        # needs dim many bits. 
        # in the first iteration we select all corners that have a 1 in the last
        # bit, i.e. select all corners that constitute a face of the dim-dimensional
        # hypercube along the last dimension.
        # the ceils mask is along this dimension in positive (i.e. rounding up) 
        # direction, whereas the floors mask is along this dimension in negative 
        # (i.e. rounding down) direction
        _hyperface_mask_ceils = (_corner_idx & _num_corners_hyperface).astype(bool)
        _hyperface_mask_floors = ~_hyperface_mask_ceils

        # now convert the boolean mask into array indices
        _hyperface_corner_idx_floors = np.argwhere(_hyperface_mask_floors).ravel()
        _hyperface_corner_idx_ceils  = np.argwhere(_hyperface_mask_ceils).ravel()

        dot_products_floors = dot_products[_hyperface_corner_idx_floors]
        dot_products_ceils  = dot_products[_hyperface_corner_idx_ceils]
        dot_products_diff   = dot_products_ceils - dot_products_floors

        # the index, dim % n, is there 
        # in the 2D case, the floors consist of the corners 0 and 1, 
        # which is top-left and top-right,
        # while the ceils consist of the opposite corners, i.e. 2, 3, aka
        # bottom-left, bottom-right.
        # now when computing their difference we interpolate along the first
        # dimension although dim corresponds to the second dimension.
        # therefore it is exactly opposite
        # (i.e. dim = 2 -> idx = 0 and dim = 1 -> idx = 1), which is exactly 
        # achieved with the modulo operator
        #dot_products = dot_products_floors + smooth_mesh_fractions[dim % n] * dot_products_diff
        dot_products = dot_products_floors + smooth_mesh_fractions[i] * dot_products_diff

    # reshape from (1, ...) to (...)
    dot_products = dot_products.squeeze()

    if n == 1:
        import matplotlib.colors as colors
        norm = colors.Normalize(dot_products.min(), dot_products.max())
        plt.scatter(mesh[0], dot_products, c=dot_products, cmap='bwr', norm=norm)
        plt.plot(mesh[0], dot_products, color='black', linewidth=0.5)
        mesh_grid = np.meshgrid(*list(map(np.arange, grid)))
        plt.scatter(mesh_grid, gradients_grid, color='black', marker='*')

    elif n == 2:
        contourf_kwargs = dict(
            levels=100, cmap='bwr'
        )
        plt.contourf(*mesh, dot_products, **contourf_kwargs)

        # show gradients in plot
        # mesh_grid = np.meshgrid(*list(map(np.arange, grid)))
        # arrow_style = dict(
        #     scale=2, scale_units='inches',
        #     width=0.005, headwidth=2,
        #     headlength=2, headaxislength=2,
        #     color='#2b2a33'
        # )
        # plt.quiver(*mesh_grid, *gradients_grid, **arrow_style)
    elif n == 3:
        import matplotlib.animation as animation

        contourf_kwargs = dict(
            levels=100, cmap='bwr'
        )
        _mesh = mesh[:2, :, :, 0]
        fig, ax = plt.subplots()
        contf = ax.contourf(*_mesh, dot_products[:, :, 0], **contourf_kwargs)

        def update(frame):
            # TODO encapsulate into function then use nonlocal instead of global
            global contf
            global resolution
            frame = frame % resolution[-1]
            contf.remove() # removes all contourf artists
            contf = ax.contourf(*_mesh, dot_products[:, :, frame], **contourf_kwargs)
            return contf

        anim = animation.FuncAnimation(fig, update, frames=resolution[-1], repeat=True)

    else:
        print('dont know yet how to plot')
    plt.show()
