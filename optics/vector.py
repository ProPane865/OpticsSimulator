import numpy as np

def normalize(vector):
    v = np.array(vector, dtype=float)
    norm = np.linalg.norm(v)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Vector must be finite and nonzero")
    
    return v / norm

def rotation_from_z(direction):
    original = np.array([0, 0, 1])
    direction_norm = normalize(direction)
    axis = np.cross(original, direction_norm)
    axis_norm = np.linalg.norm(axis)
    axis_inner = np.inner(original, direction_norm)

    if axis_norm < 1e-12:
        if axis_inner < 0:
            r_matrix = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ])
        else:
            r_matrix = np.eye(3, 3)
    else:
        axis_mat = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        r_matrix = np.eye(3, 3) + axis_mat + np.matmul(axis_mat, axis_mat) * ((1 - axis_inner) / axis_norm**2)

    return r_matrix