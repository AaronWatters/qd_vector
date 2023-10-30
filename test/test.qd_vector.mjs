import { test } from "node:test";
import assert, { deepEqual } from 'node:assert';
import * as qd_vector from '../lib/index.js';

test("will pass", () => {
    assert.strictEqual(1, 1);
});

test("check name", () => {
    assert.strictEqual(qd_vector.name, "qd_vector");
});

test("zero vector", () => {
    assert.deepEqual(qd_vector.v_zero(4), [0,0,0,0]);
});

test("v_add", () => {
    const v1 = [1,2,-3];
    const v2 = [4,-5,6];
    const sum = [5,-3,3];
    assert.deepEqual(qd_vector.v_add(v1, v2), sum);
})

test("v_scale", () => {
    const v = [1,2,-3];
    const s = 2;
    const sv = [2,4,-6];
    assert.deepEqual(qd_vector.v_scale(s, v), sv);
})

test("zero matrix", () => {
    assert.deepEqual(qd_vector.M_zero(2,3), [[0,0,0],[0,0,0]]);
});

test("matrix shape check", () => {
    const Mbad = [[0,0,0,1],[0,0,0]];
    assert.throws(() => qd_vector.M_shape(Mbad, true));
});

test("matrix shape", () => {
    const M23 = qd_vector.M_zero(2,3);
    assert.deepEqual(qd_vector.M_shape(M23, true), [2, 3]);
});

test("identity matrix", () => {
    assert.deepEqual(qd_vector.eye(2), [[1,0],[0,1]]);
});

test("swap rows", () => {
    const M = [[1,2],[3,4],[5,6]];
    const M2 = [[1,2],[5,6],[3,4]];
    assert.deepEqual(qd_vector.swap_rows(M,2,1,false), M2);
});

test("Mv_product", () => {
    const M = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]];
    const v = [4,5,-6];
    const Mv = [-7, 2, 11, 20, 29]
    assert.deepEqual(qd_vector.Mv_product(M, v), Mv);
});

test("MM_product", () => {
    const A23 = [[0, 1, 2], [3, 4, 5]];
    const A32 = [[0, 1], [2, 3], [4, 5]];
    const P = [[10, 13], [28, 40]];
    assert.deepEqual(qd_vector.MM_product(A23, A32), P)
})

test("shelf", () => {
    const M1 = [
        [1,2],
        [3,4],
    ];
    const M2 = [
        [5,6,7],
        [8,9,10],
    ];
    const M1M2 = [
        [1,2,5,6,7],
        [3,4,8,9,10]
    ];
    assert.deepEqual(qd_vector.shelf(M1, M2), M1M2)
});

test("slice", () => {
    const M = [
        [1,2,5,6,7],
        [3,4,8,9,10],
        [1,2,1,2,1]
    ];
    const Mt = [
        [1,2,5],
        [3,4,8],
    ];
    assert.deepEqual(qd_vector.M_slice(M, 0, 0, 2, 3), Mt)
});

test("reduce", () => {
    const M = [
        [1,0,1,1,0,0],
        [0,1,0,0,1,0],
        [1,2,2,0,0,1],
    ];
    const red = [
        [ 1, 0, 0, 2, 2, -1 ], 
        [ 0, 1, 0, 0, 1, 0 ], 
        [ 0, 0, 1, -1, -2, 1 ]
    ];
    assert.deepEqual(qd_vector.M_reduce(M), red);
})

test("inverse", () => {
    const M = [
        [1,0,1],
        [0,1,0],
        [1,2,2],
    ];
    const inv_exp = [
        [ 2, 2, -1 ], 
        [ 0, 1, 0 ], 
        [ -1, -2, 1 ]
    ];
    const inv = qd_vector.M_inverse(M)
    assert.deepEqual(inv, inv_exp);
    const I = qd_vector.eye(3);
    assert.deepEqual(qd_vector.MM_product(inv, M), I)
    assert.deepEqual(qd_vector.MM_product(M, inv), I)
    //assert deepEqual(qd_vector.M_)
});

test("tolerate", () => {
    assert.deepEqual(qd_vector.M_tolerate(
        [[0.000001, -0.9999999]]
    ),
    [[0, -1]])
});

test("roll 90", () => {
    const rM = qd_vector.M_tolerate(qd_vector.M_roll(Math.PI/2.0));
    const eM = [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ];
    assert.deepEqual(rM, eM);
});

test("pitch 90", () => {
    const rM = qd_vector.M_tolerate(qd_vector.M_pitch(Math.PI/2.0));
    const eM = [ 
        [ 0, 0, 1 ], 
        [ 0, 1, 0 ], 
        [ -1, 0, 0 ] 
    ];
    assert.deepEqual(rM, eM);
});

test("yaw 90", () => {
    const rM = qd_vector.M_tolerate(qd_vector.M_yaw(Math.PI/2.0));
    const eM = [ 
        [ 1, 0, 0 ], 
        [ 0, 0, 1 ], 
        [ 0, -1, 0 ] 
    ];
    assert.deepEqual(rM, eM);
});

test("v_dot", () => {
    const v1 = [1,2,3];
    const v2 = [4,5,6];
    const v1dotv2 = qd_vector.v_dot(v1, v2)
    assert.deepEqual(v1dotv2, 32);
});

test("v_cross", () => {
    const v1 = [1,2,3];
    const v2 = [4,5,6];
    const v1dotv2 = qd_vector.v_cross(v1, v2)
    assert.deepEqual(v1dotv2, [-3, 6, -3]);
});

test("M_row_major_order", () => {
    const M = [
        [1,2,3],
        [4,5,6]
    ];
    const rme = [1,2,3,4,5,6];
    const rmc = qd_vector.M_row_major_order(M)
    assert.deepEqual(rme, rmc);
});

test("M_column_major_order", () => {
    const M = [
        [1,2,3],
        [4,5,6]
    ];
    const rme = [1,4,2,5,3,6];
    const rmc = qd_vector.M_column_major_order(M)
    assert.deepEqual(rme, rmc);
});

test("M_transpose", () => {
    const M = [
        [1,2,3],
        [4,5,6]
    ];
    const rme = [[1,4],[2,5],[3,6]];
    const rmc = qd_vector.M_transpose(M)
    assert.deepEqual(rme, rmc);
});

/*
test("will fail", () => {
  throw new Error("fail");
});
*/
