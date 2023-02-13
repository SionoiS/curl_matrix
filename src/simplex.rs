#![allow(clippy::many_single_char_names)]

/*A speed-improved simplex noise algorithm for 4D.
*
* Based on example code by Stefan Gustavson (stegu@itn.liu.se).
* Optimisations by Peter Eastman (peastman@drizzle.stanford.edu).
* Better rank ordering method for 4D by Stefan Gustavson in 2012.
*
* This could be speeded up even further, but it's useful as it is.
*
* Version 2012-03-09
*
* This code was placed in the public domain by its original author,
* Stefan Gustavson. You may use it as you see fit, but
* attribution is appreciated.
* https://github.com/stegu/perlin-noise
*
* Modified by SionoiS 2019 Calculate derivatives
* Implemeted in rust by SionoiS 2020
*/

use nalgebra::{Point3, Vector3};

pub fn with_derivatives_3d(position: &Point3<f64>) -> (f64, Vector3<f64>) {
    let mut offsets = [Vector3::zeros(); 4];

    let skew_factor = F3 * position.x + F3 * position.y + F3 * position.z; // Very nice and simple skew factor for 3D

    // Skew the input space to determine which simplex cell we're in
    let mut i = (position.x + skew_factor).floor() as i64;
    let mut j = (position.y + skew_factor).floor() as i64;
    let mut k = (position.z + skew_factor).floor() as i64;

    //Factor for 3D unskewing
    let unskew_factor = G3 * i as f64 + G3 * j as f64 + G3 * k as f64;

    //Unskew the cell origin back to (x,y,z) space
    let x_0 = i as f64 - unskew_factor;
    let y_0 = j as f64 - unskew_factor;
    let z_0 = k as f64 - unskew_factor;

    //The x,y,z distances from the cell origin
    offsets[0] = Vector3::new(position.x - x_0, position.y - y_0, position.z - z_0);

    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    let (i1, j1, k1, i2, j2, k2) = if offsets[0].x >= offsets[0].y {
        if offsets[0].y >= offsets[0].z {
            // X Y Z order
            (1, 0, 0, 1, 1, 0)
        } else if offsets[0].x >= offsets[0].z {
            // X Z Y order
            (1, 0, 0, 1, 0, 1)
        } else {
            // Z X Y order
            (0, 0, 1, 1, 0, 1)
        }
    } else {
        // x0<y0
        if offsets[0].y < offsets[0].z {
            // Z Y X order
            (0, 0, 1, 0, 1, 1)
        } else if offsets[0].x < offsets[0].z {
            // Y Z X order
            (0, 1, 0, 0, 1, 1)
        } else {
            // Y X Z order
            (0, 1, 0, 1, 1, 0)
        }
    };

    // Offsets for second corner in (x,y,z) coords
    offsets[1] = Vector3::new(
        offsets[0].x - i1 as f64 + G3,
        offsets[0].y - j1 as f64 + G3,
        offsets[0].z - k1 as f64 + G3,
    );

    // Offsets for third corner in (x,y,z) coords
    offsets[2] = Vector3::new(
        offsets[0].x - i2 as f64 + 2.0 * G3,
        offsets[0].y - j2 as f64 + 2.0 * G3,
        offsets[0].z - k2 as f64 + 2.0 * G3,
    );

    // Offsets for fourth corner in (x,y,z) coords
    offsets[3] = Vector3::new(
        offsets[0].x - 1.0 + 3.0 * G3,
        offsets[0].y - 1.0 + 3.0 * G3,
        offsets[0].z - 1.0 + 3.0 * G3,
    );

    // Work out the hashed gradient indices of the five simplex corners
    i &= 0xFF;
    j &= 0xFF;
    k &= 0xFF;

    let indices_i = [i, i + i1, i + i2, i + 1];
    let indices_j = [j, j + j1, j + j2, j + 1];
    let indices_k = [k, k + k1, k + k2, k + 1];

    let mut n = 0.0;
    let mut derivatives = Vector3::zeros();

    for (i, offset) in offsets.iter().enumerate() {
        let t = 0.5 - offset.dot(&offset);

        if t < 0.0 {
            continue;
        }

        let t2 = t * t;
        let t4 = t2 * t2;

        let gradient = GRADIANTS_3D[(SEED[indices_i[i] as usize
            + SEED[indices_j[i] as usize + SEED[indices_k[i] as usize] as usize] as usize]
            % 12) as usize];

        let grad_dot = gradient.dot(&offset);

        n += t4 * grad_dot;

        derivatives += -8.0 * t2 * offset * grad_dot + t4 * gradient;
    }

    (n * 72.0, derivatives * 72.0)
}

// Skewing and unskewing factors
const F3: f64 = 0.333_333_333_333_333_3; // 1.0 / 3.0
const G3: f64 = 0.166_666_666_666_666_66; // 1.0 / 6.0

const GRADIANTS_3D: [Vector3<f64>; 12] = [
    Vector3::new(1.0, 1.0, 0.0),
    Vector3::new(-1.0, 1.0, 0.0),
    Vector3::new(1.0, -1.0, 0.0),
    Vector3::new(-1.0, -1.0, 0.0),
    Vector3::new(1.0, 0.0, 1.0),
    Vector3::new(-1.0, 0.0, 1.0),
    Vector3::new(1.0, 0.0, -1.0),
    Vector3::new(-1.0, 0.0, -1.0),
    Vector3::new(0.0, 1.0, 1.0),
    Vector3::new(0.0, -1.0, 1.0),
    Vector3::new(0.0, 1.0, -1.0),
    Vector3::new(0.0, -1.0, -1.0),
];

pub const SEED: [u8; 512] = [
    210, 251, 147, 139, 214, 27, 149, 231, 162, 19, 136, 158, 232, 78, 82, 140, 37, 208, 50, 73,
    79, 79, 240, 100, 144, 14, 172, 250, 59, 61, 226, 229, 69, 197, 143, 251, 125, 115, 197, 14,
    102, 150, 63, 90, 157, 224, 161, 42, 42, 30, 183, 133, 168, 157, 150, 206, 221, 140, 70, 192,
    153, 25, 7, 167, 9, 246, 218, 174, 99, 134, 163, 46, 38, 189, 228, 223, 54, 147, 16, 144, 213,
    83, 59, 156, 31, 1, 80, 132, 0, 182, 205, 177, 79, 77, 230, 153, 109, 231, 185, 24, 253, 191,
    193, 13, 2, 86, 95, 118, 181, 161, 179, 129, 203, 23, 170, 111, 174, 225, 188, 166, 123, 12,
    163, 123, 206, 225, 80, 194, 191, 98, 248, 239, 155, 8, 102, 239, 133, 94, 194, 134, 42, 118,
    102, 56, 28, 219, 202, 219, 150, 200, 3, 195, 36, 127, 57, 219, 179, 150, 75, 64, 148, 153,
    126, 240, 121, 210, 216, 5, 149, 205, 10, 160, 247, 191, 137, 139, 210, 181, 189, 85, 237, 145,
    75, 77, 97, 97, 181, 143, 93, 151, 166, 8, 176, 97, 182, 14, 126, 38, 187, 145, 23, 239, 64,
    55, 203, 45, 25, 8, 237, 122, 43, 16, 17, 20, 216, 6, 31, 202, 232, 133, 163, 56, 210, 81, 169,
    252, 245, 38, 160, 198, 172, 165, 234, 78, 77, 96, 32, 58, 126, 196, 117, 140, 247, 94, 203,
    166, 232, 198, 143, 247, 126, 175, 42, 21, 185, 70, 210, 251, 147, 139, 214, 27, 149, 231, 162,
    19, 136, 158, 232, 78, 82, 140, 37, 208, 50, 73, 79, 79, 240, 100, 144, 14, 172, 250, 59, 61,
    226, 229, 69, 197, 143, 251, 125, 115, 197, 14, 102, 150, 63, 90, 157, 224, 161, 42, 42, 30,
    183, 133, 168, 157, 150, 206, 221, 140, 70, 192, 153, 25, 7, 167, 9, 246, 218, 174, 99, 134,
    163, 46, 38, 189, 228, 223, 54, 147, 16, 144, 213, 83, 59, 156, 31, 1, 80, 132, 0, 182, 205,
    177, 79, 77, 230, 153, 109, 231, 185, 24, 253, 191, 193, 13, 2, 86, 95, 118, 181, 161, 179,
    129, 203, 23, 170, 111, 174, 225, 188, 166, 123, 12, 163, 123, 206, 225, 80, 194, 191, 98, 248,
    239, 155, 8, 102, 239, 133, 94, 194, 134, 42, 118, 102, 56, 28, 219, 202, 219, 150, 200, 3,
    195, 36, 127, 57, 219, 179, 150, 75, 64, 148, 153, 126, 240, 121, 210, 216, 5, 149, 205, 10,
    160, 247, 191, 137, 139, 210, 181, 189, 85, 237, 145, 75, 77, 97, 97, 181, 143, 93, 151, 166,
    8, 176, 97, 182, 14, 126, 38, 187, 145, 23, 239, 64, 55, 203, 45, 25, 8, 237, 122, 43, 16, 17,
    20, 216, 6, 31, 202, 232, 133, 163, 56, 210, 81, 169, 252, 245, 38, 160, 198, 172, 165, 234,
    78, 77, 96, 32, 58, 126, 196, 117, 140, 247, 94, 203, 166, 232, 198, 143, 247, 126, 175, 42,
    21, 185, 70,
];
