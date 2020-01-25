import numpy as np


class Vector:
    """Vector Class.

    A helper class to make easier to work with Vectors in
    future projects. Can be generalized to more dimensions.
    """

    def __init__(self, coords):
        self.coords = np.array(coords)
        # Set x and y for convenience in 2D
        if len(self.coords) == 2:
            self.x = coords[0]
            self.y = coords[1]

    def __sub__(self, other):
        return Vector(np.subtract(self.coords, other.coords))

    def __add__(self, other):
        return Vector(np.add(self.coords, other.coords))

    def __mul__(self, other):
        # works for Vector * Vector and Vector * Scalar
        if isinstance(other, Vector):
            return Vector(np.multiply(self.coords, other.coords))
        else:
            return Vector(np.multiply(self.coords, other))

    def __str__(self):
        if len(self.coords) == 2:
            return f'({self.x}, {self.y})'
        else:
            return f'Vector([{self.coords}])'

    def __repr__(self):
        return self.__str__()

    def magnitude(self):
        return np.sqrt(np.sum(np.square(self.coords)))

    def normalized(self):
        assert self.magnitude != 0
        return Vector(np.divide(self.coords, self.magnitude()))

    def perpendicular(self):
        # Works for 2D only
        assert len(self.coords) == 2, 'Perpendicular only works in 2D'

        # If this is a zero vector, return self
        if not any(self.coords):
            return self

        return Vector((self.y, -self.x))
