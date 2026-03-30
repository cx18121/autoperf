import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row of a matrix to have unit L2 norm.

    Args:
        matrix: 2D numpy array of shape (n_rows, n_cols)

    Returns:
        A new array of the same shape where each row has L2 norm == 1.
        Rows with zero norm are left as zeros.
    """
    n_rows, n_cols = matrix.shape
    result = np.empty_like(matrix)

    for i in range(n_rows):
        # Compute L2 norm for this row using a manual loop
        norm_sq = 0.0
        for j in range(n_cols):
            norm_sq += matrix[i, j] * matrix[i, j]
        norm = norm_sq ** 0.5

        # Normalize the row element-by-element
        if norm > 0:
            for j in range(n_cols):
                result[i, j] = matrix[i, j] / norm
        else:
            for j in range(n_cols):
                result[i, j] = 0.0

    return result
