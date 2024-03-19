import warnings

import numpy as np

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.tools import extend_docstring


class _SphereBase(RiemannianSubmanifold):
    def __init__(self, *shape, name, dimension):
        if len(shape) == 0:
            raise TypeError("Need at least one dimension.")
        self._shape = shape
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return np.pi

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    def dist(self, point_a, point_b):
        inner = max(min(self.inner_product(point_a, point_a, point_b), 1), -1)
        return np.arccos(inner)

    def projection(self, point, vector):
        return vector - self.inner_product(point, point, vector) * point

    to_tangent_space = projection

    def weingarten(self, point, tangent_vector, normal_vector):
        return (
            -self.inner_product(point, point, normal_vector) * tangent_vector
        )

    def exp(self, point, tangent_vector):
        norm = self.norm(point, tangent_vector)
        return point * np.cos(norm) + tangent_vector * np.sinc(norm / np.pi)

    def retraction(self, point, tangent_vector):
        return self._normalize(point + tangent_vector)

    def log(self, point_a, point_b):
        vector = self.projection(point_a, point_b - point_a)
        distance = self.dist(point_a, point_b)
        epsilon = np.finfo(np.float64).eps
        factor = (distance + epsilon) / (self.norm(point_a, vector) + epsilon)
        return factor * vector

    def random_point(self):
        point = np.random.normal(size=self._shape)
        return self._normalize(point)

    def random_tangent_vector(self, point):
        vector = np.random.normal(size=self._shape)
        return self._normalize(self.projection(point, vector))

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def pair_mean(self, point_a, point_b):
        return self._normalize(point_a + point_b)

    def zero_vector(self, point):
        return np.zeros(self._shape)

    def _normalize(self, array):
        return array / np.linalg.norm(array)


DOCSTRING_NOTE = """
    Note:
        The Weingarten map is taken from [AMT2013]_.
"""


@extend_docstring(DOCSTRING_NOTE)
class Sphere(_SphereBase):
    r"""The sphere manifold.

    Manifold of shape :math:`n_1 \times \ldots \times n_k` tensors with unit
    Euclidean norm.
    The norm is understood as the :math:`\ell_2`-norm of :math:`\E =
    \R^{\sum_{i=1}^k n_i}` after identifying :math:`\R^{n_1 \times \ldots
    \times n_k}` with :math:`\E`.
    The metric is the one inherited from the usual Euclidean inner product that
    induces :math:`\norm{\cdot}_2` on :math:`\E` such that the manifold forms a
    Riemannian submanifold of Euclidean space.

    Args:
        shape: The shape of tensors.
    """

    def __init__(self, *shape: int):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        if len(shape) == 1:
            (n,) = shape
            name = f"Sphere manifold of {n}-vectors"
        elif len(shape) == 2:
            m, n = shape
            name = f"Sphere manifold of {m}x{n} matrices"
        else:
            name = f"Sphere manifold of shape {shape} tensors"
        dimension = np.prod(shape) - 1
        super().__init__(*shape, name=name, dimension=dimension)


class _SphereSubspaceIntersectionManifold(_SphereBase):
    def __init__(self, projector, name, dimension):
        m, n = projector.shape
        assert m == n, "projection matrix is not square"
        if dimension == 0:
            warnings.warn(
                "Intersected subspace is 1-dimensional. The manifold "
                "therefore has dimension 0 as it only consists of isolated "
                "points"
            )
        self._subspace_projector = projector
        super().__init__(n, name=name, dimension=dimension)

    def _validate_span_matrix(self, matrix):
        if len(matrix.shape) != 2:
            raise ValueError("Input array must be 2-dimensional")
        num_rows, num_columns = matrix.shape
        if num_rows < num_columns:
            raise ValueError(
                "The span matrix cannot have fewer rows than columns"
            )

    def projection(self, point, vector):
        return self._subspace_projector @ super().projection(point, vector)

    def random_point(self):
        point = super().random_point()
        return self._normalize(self._subspace_projector @ point)

    def random_tangent_vector(self, point):
        vector = super().random_tangent_vector(point)
        return self._normalize(self._subspace_projector @ vector)


@extend_docstring(DOCSTRING_NOTE)
class SphereSubspaceIntersection(_SphereSubspaceIntersectionManifold):
    r"""Sphere-subspace intersection manifold.

    Manifold of :math:`n`-dimensional vectors with unit :math:`\ell_2`-norm
    intersecting an :math:`r`-dimensional subspace of :math:`\R^n`.
    The subspace is represented by a matrix of size ``n x r`` whose columns
    span the subspace.

    Args:
        matrix: Matrix whose columns span the intersecting subspace.
    """

    def __init__(self, matrix):
        self._validate_span_matrix(matrix)
        m = matrix.shape[0]
        q, _ = np.linalg.qr(matrix)
        projector = q @ q.T
        subspace_dimension = np.linalg.matrix_rank(projector)
        name = (
            f"Sphere manifold of {m}-dimensional vectors intersecting a "
            f"{subspace_dimension}-dimensional subspace"
        )
        dimension = subspace_dimension - 1
        super().__init__(projector, name, dimension)


@extend_docstring(DOCSTRING_NOTE)
class SphereSubspaceComplementIntersection(
    _SphereSubspaceIntersectionManifold
):
    r"""Sphere-subspace complement intersection manifold.

    Manifold of :math:`n`-dimensional vectors with unit :math:`\ell_2`-norm
    that are orthogonal to an :math:`r`-dimensional subspace of :math:`\R^n`.
    The subspace is represented by a matrix of size ``n x r`` whose columns
    span the subspace.

    Args:
        matrix: Matrix whose columns span the subspace.
    """

    def __init__(self, matrix):
        self._validate_span_matrix(matrix)
        m = matrix.shape[0]
        q, _ = np.linalg.qr(matrix)
        projector = np.eye(m) - q @ q.T
        subspace_dimension = np.linalg.matrix_rank(projector)
        name = (
            f"Sphere manifold of {m}-dimensional vectors orthogonal "
            f"to a {subspace_dimension}-dimensional subspace"
        )
        dimension = subspace_dimension - 1
        super().__init__(projector, name, dimension)



class ComplementBall(RiemannianSubmanifold):
    def __init__(self, m: int, n: int, min_norm: np.ndarray, tol: float):
        self._m = m
        self._n = n
        self._min_norm = min_norm
        self._tol = tol
        self._shape = (m,n)
        name = f"Product of complement of ball"
        dimension = m * n
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return np.pi * np.sqrt(self._n)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return float(
            np.real(
                np.tensordot(
                    tangent_vector_a.conj(),
                    tangent_vector_b,
                    axes=tangent_vector_a.ndim,
                )
            )
        )
    
    # To be modified ??
    def weingarten(self, point, tangent_vector, normal_vector):
        return np.zeros_like(tangent_vector)

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    def dist(self, point_a, point_b):
        return np.linalg.norm(point_a - point_b)

    def _projection(self, point, vector, _min):
        norm = np.linalg.norm(point)
        if norm > _min:
            return vector
        else:
            inner = np.dot(vector,point)
            if inner > 0:
                return vector
            else:
                return vector - inner*point/norm
    
    def projection(self, point, vector):
        return np.array([self._projection(p, v, bound) for p, v, bound in zip(point.transpose(),vector.transpose(),self._min_norm)]).transpose()
    
    to_tangent_space = projection

    def _exp(self, point, tangent_vector, _min):
        norm = np.linalg.norm(point)
        if norm > _min:
            return point + tangent_vector
        else:
            inner = np.dot(tangent_vector,point)
            if inner > 0:
                return point + tangent_vector
            else:
                norm = self.norm(point, tangent_vector)
                return point * np.cos(norm) + tangent_vector * np.sinc(norm / np.pi)

    
    def exp(self, point, tangent_vector):
        return np.array([self._exp(p, v, bound) for p, v, bound in zip(point.transpose(),tangent_vector.transpose(),self._min_norm)]).transpose()
    
    def _retraction(self, point, tangent_vector, _min):
        norm = np.linalg.norm(point)
        if norm > _min:
            return point + tangent_vector
        else:
            inner = np.dot(tangent_vector,point)
            if inner > 0:
                return point + tangent_vector
            else:
                norm = self.norm(point, tangent_vector)
                return self._normalize(point + tangent_vector)*_min
            
    def retraction(self, point, tangent_vector):
        return np.array([self._retraction(p, v, bound) for p, v, bound in zip(point.transpose(),tangent_vector.transpose(),self._min_norm)]).transpose()

    def _log(self, point_a, point_b):
        norm = np.linalg.norm(point_a)
        if norm > self._min:
            return point_b - point_a
        else:
            inner = np.dot(point_b - point_a,point_a)
            if inner > 0:
                return point_b - point_a
            else:
                return point_b - point_a - inner*point_a/norm
    
    def log(self, point, tangent_vector):
        return np.array([self._log(p, v, bound) for p, v, bound in zip(point.transpose(),tangent_vector.transpose(),self._min_norm)]).transpose()

    def random_point(self):
        v = np.random.normal(size=self._shape)
        norm = np.linalg.norm(v,axis=0)
        v = (norm + self._min_norm)*v
        return v

    def random_tangent_vector(self, point):
        tangent_vector = self.random_point()
        return tangent_vector / self.norm(point, tangent_vector)

    def _transport(self, point_a, point_b, tangent_vector_a,_min):
        norm = np.linalg.norm(point_a)
        if norm > _min:
            return tangent_vector_a
        else:
            inner = np.dot(tangent_vector_a,point_b)
            if inner > 0:
                return tangent_vector_a
            else:
                return tangent_vector_a - inner*point_b/np.linalg.norm(point_b)
    
    def transport(self, point_a, point_b, tangent_vector_a):
        return np.array([self._transport(p1, p2, v, bound) for p1, p2, v, bound in zip(point_a.transpose(),point_b.transpose(),tangent_vector_a.transpose(),self._min_norm)]).transpose()

    def _pair_mean(self, point_a, point_b, _min):
        v = (point_a + point_b) / 2
        norm = np.linalg.norm(v)
        if norm < _min:
            return self._normalize(v)*_min
        else: 
            return v
        
    def pair_mean(self, point_a, point_b):
        return np.array([self._pair_mean(p, v, bound) for p, v, bound in zip(point_a.transpose(),point_b.transpose(),self._min_norm)]).transpose()


    def zero_vector(self, point):
        return np.zeros(self._shape)
    
    def _normalize(self, array):
        return array / np.linalg.norm(array)
    

class Ball(RiemannianSubmanifold):
    def __init__(self, m: int, n: int, max_norm: np.ndarray, tol: float):
        self._m = m
        self._n = n
        self._max_norm = max_norm
        self._tol = tol
        self._shape = (m,n)
        name = f"Product of complement of ball"
        dimension = m * n
        super().__init__(name, dimension)

    def typical_dist(self):
        return np.pi * np.sqrt(self._n)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return float(
            np.real(
                np.tensordot(
                    tangent_vector_a.conj(),
                    tangent_vector_b,
                    axes=tangent_vector_a.ndim,
                )
            )
        )
    
    # To be modified ??
    def weingarten(self, point, tangent_vector, normal_vector):
        return np.zeros_like(tangent_vector)

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    def dist(self, point_a, point_b):
        return np.linalg.norm(point_a - point_b)

    def _projection(self, point, vector, _max):
        norm = np.linalg.norm(point)
        if norm < _max:
            return vector
        else:
            inner = np.dot(vector,point)
            if inner < 0:
                return vector
            else:
                return vector - inner*point/norm
    
    def projection(self, point, vector):
        return np.array([self._projection(p, v, bound) for p, v, bound in zip(point.transpose(),vector.transpose(),self._max_norm)])
    
    to_tangent_space = projection

    def _exp(self, point, tangent_vector, _max):
        norm = np.linalg.norm(point)
        if norm < _max:
            return point + tangent_vector
        else:
            inner = np.dot(tangent_vector,point)
            if inner < 0:
                return point + tangent_vector
            else:
                norm = self.norm(point, tangent_vector)
                return point * np.cos(norm) + tangent_vector * np.sinc(norm / np.pi)

    
    def exp(self, point, tangent_vector):
        return np.array([self._exp(p, v, bound) for p, v, bound in zip(point.transpose(),tangent_vector.transpose(),self._max_norm)])
    
    def _retraction(self, point, tangent_vector, _max):
        norm = np.linalg.norm(point)
        if norm < _max:
            return point + tangent_vector
        else:
            inner = np.dot(tangent_vector,point)
            if inner < 0:
                return point + tangent_vector
            else:
                norm = self.norm(point, tangent_vector)
                return self._normalize(point + tangent_vector)*_max
            
    def retraction(self, point, tangent_vector):
        return np.array([self._retraction(p, v, bound) for p, v, bound in zip(point.transpose(),tangent_vector.transpose(),self._max_norm)])

    def _log(self, point_a, point_b):
        norm = np.linalg.norm(point_a)
        if norm < self._max:
            return point_b - point_a
        else:
            inner = np.dot(point_b - point_a,point_a)
            if inner < 0:
                return point_b - point_a
            else:
                return point_b - point_a - inner*point_a/norm
    
    def log(self, point, tangent_vector):
        return np.array([self._log(p, v, bound) for p, v, bound in zip(point.transpose(),tangent_vector.transpose(),self._max_norm)])

    def random_point(self):
        v = np.random.normal(size=self._shape)
        return v

    def random_tangent_vector(self, point):
        tangent_vector = self.random_point()
        return tangent_vector / self.norm(point, tangent_vector)

    def _transport(self, point_a, point_b, tangent_vector_a,_max):
        norm = np.linalg.norm(point_a)
        if norm < _max:
            return tangent_vector_a
        else:
            inner = np.dot(tangent_vector_a,point_b)
            if inner < 0:
                return tangent_vector_a
            else:
                return tangent_vector_a - inner*point_b/np.linalg.norm(point_b)
    
    def transport(self, point_a, point_b, tangent_vector_a):
        return np.array([self._transport(p1, p2, v, bound) for p1, p2, v, bound in zip(point_a.transpose(),point_b.transpose(),tangent_vector_a.transpose(),self._max_norm)])

    def _pair_mean(self, point_a, point_b, _max):
        v = (point_a + point_b) / 2
        norm = np.linalg.norm(v)
        if norm < _max:
            return self._normalize(v)*_max
        else: 
            return v
        
    def pair_mean(self, point_a, point_b):
        return np.array([self._pair_mean(p, v, bound) for p, v, bound in zip(point_a.transpose(),point_b.transpose(),self._max_norm)])


    def zero_vector(self, point):
        return np.zeros(self._shape)
    
    def _normalize(self, array):
        return array / np.linalg.norm(array)
