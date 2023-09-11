
export const name = "qd_vector";

/** Make a zero vector of length n  */
export function v_zero(n) {
    const b = new Float64Array(n);
    return Array.from(b);
};

export function v_add(v1, v2) {
    const N = v1.length;
    const result = v_zero(N);
    for (var i=0; i<N; i++) {
        result[i] = v1[i] + v2[i];
    }
    return result;
};

export function v_scale(s, v) {
    const N = v.length;
    const result = v_zero(N);
    for (var i=0; i<N; i++) {
        result[i] = s * v[i];
    }
    return result;
};

/** Make a zero matrix, n rows, m columns */
export function M_zero(n, m) {
    const result = [];
    for (var i=0; i<n; i++) {
        result.push(v_zero(m));
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

export function M_inverse(M) {
    const dim = M.length;
    const I = eye(dim);
    const Mext = shelf(M, I);
    const red = M_reduce(Mext);
    const inv = M_slice(red, 0, dim, dim, 2 * dim);
    return inv;
};