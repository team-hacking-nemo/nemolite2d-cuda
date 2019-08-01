# dl_esm_inf
Earth-System Modelling Infrastructure library.

A library to aid the creation of earth-system models. Currently
supports two-dimensional, finite-difference models.

The first version of this library was developed to support 2D finite-
difference shallow-water models in the GOcean Project.

## Concepts ##

dl_esm_inf provides certain types of object from which an earth-system
model may then be constructed

### Grid ###

The grid of points making up the simulation domain.

### Field ###

A field is a representation of some physical quantity (e.g. vorticity)
at points on the grid.
