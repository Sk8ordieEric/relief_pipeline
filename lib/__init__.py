"""Shared building blocks for the CNC relief and inlay pipeline.

    tooling   the cutter's real geometry, and what it can and cannot reach
    flow      coherence-enhancing diffusion (fur, feathers, hair)
    fields    sub-pixel distance fields, Poisson integration, image loading
    mesh      watertight grid meshes, verification, binary STL output

Every length in this package is MILLIMETRES.
"""
__all__ = ["tooling", "flow", "fields", "mesh"]
