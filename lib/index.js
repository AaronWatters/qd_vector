
export const name = "qd_vector";

/** Make a zero vector of length n  */
export function v_zero(n) {
    const b = new Float64Array(n);
    return Array.from(b);
};

/** Add two vectors */
export function v_add(v1, v2) {
    const N = v1.length;
    const result = v_zero(N);
    for (var i=0; i<N; i++) {
        result[i] = v1[i] + v2[i];
    }
    return result;
};

/** Pointwise vector minimum */
export function v_minimum(v1, v2) {
    const N = v1.length;
    const result = v_zero(N);
    for (var i=0; i<N; i++) {
        result[i] = Math.min(v1[i], v2[i]);
    }
    return result;
};

/** Pointwise vector maximum */
export function v_maximum(v1, v2) {
    const N = v1.length;
    const result = v_zero(N);
    for (var i=0; i<N; i++) {
        result[i] = Math.max(v1[i], v2[i]);
    }
    return result;
};

/** multiply a scalar and a vector */
export function v_scale(s, v) {
    const N = v.length;
    const result = v_zero(N);
    for (var i=0; i<N; i++) {
        result[i] = s * v[i];
    }
    return result;
};

/** Subtract two vectors */
export function v_sub(v1, v2) {
    return v_add(
        v1,
        v_scale(-1, v2)
    );
};

/** Euclidean vector length */
export function v_length(v) {
    var sum = 0;
    for (var member of v) {
        sum += member * member;
    }
    return Math.sqrt(sum);
};

/** Vector normalized to length 1.0 in euclidean norm. */
export function v_normalize(v) {
    var ln = v_length(v);
    return v_scale(1.0/ln, v);
};

/** Make a zero matrix, n rows, m columns */
export function M_zero(n, m) {
    const result = [];
    for (var i=0; i<n; i++) {
        result.push(v_zero(m));
    }
    return result;
};

/** Make a 3d graphics affine matrix (4x4) from a rotation (3x3) and translation vector */
export function affine3d(rotation3x3, translationv3) {
    const result = eye(4);
    // default to identity rotation
    if (rotation3x3) {
        for (var i=0; i<3; i++) {
            for (var j=0; j<3; j++) {
                result[i][j] = rotation3x3[i][j];
            }
        }
    }
    if (translationv3) {
        for (var i=0; i<3; i++) {
            result[i][3] = translationv3[i]; 
        }
    }
    return result;
};

/** Apply an affine 4x4 transform matrix for 3d space to a 3d vector */
export function apply_affine3d(affine3d, vector3d) {
    const v4 = vector3d.slice();
    v4.push(1);
    const v4transformed = Mv_product(affine3d, v4)
    const v3transformed = v4transformed.slice(0, 3);
    return v3transformed;
};

/** Flatten a matrix into a list. */
export function M_as_list(M) {
    const result = [];
    const nrows = M.length;
    for (var i=0; i<nrows; i++) {
        result.push(...M[i]);
    }
    return result;
};

export function list_as_M(L, nrows, ncols) {
    const nitems = L.length;
    if (nitems != (nrows * ncols)) {
        throw new Error(`Length ${nitems} doesn't match rows ${nrows} and columns ${ncols}.`)
    }
    const result = [];
    var cursor = 0;
    for (var i=0; i<nrows; i++) {
        const row = [];
        for (var j=0; j<ncols; j++) {
            const item = L[cursor];
            row.push(item);
            cursor ++;
        }
        result.push(row);
    }
    return result;
};

/** Get the [row, columns] shape of matrix M. */
export function M_shape(M, check) {
    const nrows = M.length;
    const ncols = M[0].length;
    if (check) {
        for (var i=0; i<nrows; i++) {
            if (M[i].length != ncols) {
                throw new Error("inconsistent shape.");
            }
        }
    }
    return [nrows, ncols];
};

/** Make an n x n identity matrix */
export function eye(n) {
    const result = M_zero(n, n);
    for (var i=0; i<n; i++) {
        result[i][i] = 1;
    }
    return result;
};

/** Matrix.dot(vector) */
export function Mv_product(M, v) {
    const [nrows, ncols] = M_shape(M);
    var result = v_zero(nrows);
    for (var i=0; i<nrows; i++) {
        var value = 0;
        for (var j=0; j<ncols; j++) {
            value += M[i][j] * v[j];
        }
        result[i] = value;
    }
    return result;
};

export function MM_product(M1, M2) {
    const [nrows1, ncols1] = M_shape(M1);
    const [nrows2, ncols2] = M_shape(M2);
    if (ncols1 != nrows2) {
        throw new Error("incompatible matrices.");
    }
    var result = M_zero(nrows1, ncols2)
    for (var i=0; i<nrows1; i++) {
        for (var j=0; j<ncols2; j++) {
            var rij = 0.0;
            for (var k=0; k<nrows2; k++) {
                //console.log(rij, M1[i][k], M2[k][j])
                rij += M1[i][k] * M2[k][j];
            }
            result[i][j] = rij;
        }
    }
    return result;
};

/** Matrix copy */
export function M_copy(M) {
    const [nrows, ncols] = M_shape(M);
    const result = M_zero(nrows, ncols);
    for (var i=0; i<nrows; i++) {
        for (var j=0; j<ncols; j++) {
            result[i][j] = M[i][j];
        }
    }
    return result;
};

/** round near integer values (mainly for testing) */
export function M_tolerate(M, epsilon=0.001) {
    const result = M_copy(M);
    for (var row of result) {
        for (var i=0; i<row.length; i++) {
            const v = row[i];
            const rv = Math.round(v)
            if (Math.abs(v - rv) < epsilon) {
                row[i] = rv;
            }
        }
    }
    return result;
}

/** Swap row i with row j from M. */
export function swap_rows(M, i, j, in_place) {
    var result = M;
    if (!in_place) {
        result = M_copy(M);
    }
    const rowi = result[i];
    result[i] = result[j];
    result[j] = rowi;
    return result;
};

/** Adjoin [M1 | M2] */
export function shelf(M1, M2) {
    const [nrows1, ncols1] = M_shape(M1);
    const [nrows2, ncols2] = M_shape(M2);
    if (nrows1 != nrows2) {
        throw new Error("bad shapes: rows must match.");
    }
    const result = M_zero(nrows1, ncols1 + ncols2);
    for (var row=0; row<nrows2; row++) {
        for (var col1=0; col1<ncols1; col1++) {
            result[row][col1] = M1[row][col1];
        }
        for (var col2=0; col2<ncols2; col2++) {
            result[row][col2 + ncols1] = M2[row][col2];
        }
    }
    return result;
};

/** Return equivalent of numpy M[minrow:maxrow, mincol:maxcol] */
export function M_slice(M, minrow, mincol, maxrow, maxcol) {
    const nrows = maxrow - minrow;
    const ncols = maxcol - mincol;
    const result = M_zero(nrows, ncols);
    for (var i=0; i<nrows; i++) {
        for (var j=0; j<ncols; j++) {
            result[i][j] = M[i+minrow][j+mincol];
        }
    }
    return result;
};

/** Row-eschelon reduction. */
export function M_reduce(M) {
    var result = M_copy(M);
    const [nrows, ncols] = M_shape(M);
    const MN = Math.min(nrows, ncols);
    for (var col=0; col<MN; col++) {
        var swaprow = col;
        var swapvalue = Math.abs(result[swaprow][col]);
        for (var row=col+1; row<MN; row++) {
            const testvalue = Math.abs(result[row][col]);
            if (testvalue > swapvalue) {
                swapvalue = testvalue;
                swaprow = row;
            }
        }
        if (swaprow != row) {
            result = swap_rows(result, col, swaprow, true);
        }
        var pivot_value = result[col][col];
        var scale = 1.0 / pivot_value;
        var pivot_row = v_scale(scale, result[col])
        for (var row=0; row<MN; row++) {
            const vrow = result[row];
            if (row == col) {
                result[row] = pivot_row;
            } else {
                const row_value = vrow[col];
                const adjust = v_scale(- row_value, pivot_row);
                const adjusted_row = v_add(vrow, adjust);
                result[row] = adjusted_row;
            }
        }
    }
    return result;
};

/** simple matrix inverse. */
export function M_inverse(M) {
    const dim = M.length;
    const I = eye(dim);
    const Mext = shelf(M, I);
    const red = M_reduce(Mext);
    const inv = M_slice(red, 0, dim, dim, 2 * dim);
    return inv;
};

/** aircraft roll matrix */
export function M_roll(roll) {
    var cr = Math.cos(roll);
    var sr = Math.sin(roll);
    var rollM = [
        [cr, -sr, 0],
        [sr, cr, 0],
        [0, 0, 1],
    ];
    return rollM;
};

/** aircraft pitch matrix */
export function M_pitch(pitch) {
    var cp = Math.cos(pitch);
    var sp = Math.sin(pitch);
    var pitchM = [
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ];
    return pitchM;
};

/** aircraft yaw matrix */
export function M_yaw(yaw) {
    var cy = Math.cos(yaw);
    var sy = Math.sin(yaw)
    var yawM = [
        [1, 0, 0],
        [0, cy, sy],
        [0, -sy, cy],
    ];
    return yawM;
};

/** vector dot product */
export function v_dot(v1, v2) {
    const n = v1.length;
    var result = 0.0;
    for (var i=0; i<n; i++) {
        result += v1[i] * v2[i]
    }
    return result;
};

/** 3d vector cross product */
export function v_cross(v1, v2) {
    // from https://en.wikipedia.org/wiki/Cross_product
    const [a1, a2, a3] = v1;
    const [b1, b2, b3] = v2;
    const result = [
        a2 * b3 - a3 * b2,
        a3 * b1 - a1 * b3,
        a1 * b2 - a2 * b1
    ];
    return result;
}

export function M_row_major_order(M) {
    var result = [];
    const [nrows, ncols] = M_shape(M);
    for (var row=0; row<nrows; row++) {
        for (var col=0; col<ncols; col++) {
            result.push(M[row][col]);
        }
    }
    return result;
};

export function M_column_major_order(M) {
    var result = [];
    const [nrows, ncols] = M_shape(M);
    for (var col=0; col<ncols; col++) {
        for (var row=0; row<nrows; row++) {
            result.push(M[row][col]);
        }
    }
    return result;
};

export function M_transpose(M) {
    var result = [];
    const [nrows, ncols] = M_shape(M);
    for (var col=0; col<ncols; col++) {
        var r = [];
        for (var row=0; row<nrows; row++) {
            r.push(M[row][col]);
        }
        result.push(r);
    }
    return result;
};
