//! Core library for curl noise particle simulation.
//!
//! Provides simplex noise generation with derivative calculation,
//! curl noise vector field generation, and particle types.

mod simplex;

pub use simplex::{with_derivatives_3d, SEED};

use nalgebra::{Point2, Point3, Vector2};
use rand::prelude::*;

/// A particle in the simulation.
#[derive(Debug, Clone)]
pub struct Particle<T> {
    /// Color of the particle (RGB).
    pub color: T,
    /// Current position of the particle.
    pub coords: Point2<f64>,
    /// Time-to-live: remaining frames before particle regeneration.
    pub ttl: usize,
}

/// Computes curl noise at 2D coordinates for a given time value.
///
/// Uses 3D simplex noise with the time dimension as the third coordinate,
/// then computes the curl to produce a divergence-free vector field.
pub fn curl_noise_2d(coordinates: &Point2<f64>, time: f64) -> Vector2<f64> {
    let space_time = Point3::new(coordinates.x, coordinates.y, time);
    let (_, deriv) = with_derivatives_3d(&space_time);
    let derivatives = &Vector2::new(deriv.x, deriv.y);
    curl_2d(derivatives)
}

/// Converts potential field derivatives to a curl vector field.
///
/// For a 2D curl, we take the y derivative as the x component
/// and the negative x derivative as the y component.
fn curl_2d(derivatives: &Vector2<f64>) -> Vector2<f64> {
    Vector2::new(derivatives.y, -derivatives.x)
}

/// Generates a vector field for the given dimensions and time.
///
/// # Arguments
/// * `width` - Width of the field in pixels
/// * `height` - Height of the field in pixels
/// * `field_scale` - Scaling factor for noise coordinates
/// * `time` - Time value for the noise function
pub fn generate_vector_field(
    width: u32,
    height: u32,
    field_scale: f64,
    time: f64,
) -> Vec<Vector2<f64>> {
    let pixel_count = width * height;
    let mut vector_field = Vec::with_capacity(pixel_count as usize);

    for y in 0..height {
        for x in 0..width {
            let coords = Point2::new(x as f64 * field_scale, y as f64 * field_scale);
            let vector = curl_noise_2d(&coords, time);
            vector_field.push(vector);
        }
    }

    vector_field
}

/// Creates a random particle with the given bounds.
///
/// # Arguments
/// * `rng` - Random number generator
/// * `width` - Maximum width for particle position
/// * `height` - Maximum height for particle position
/// * `max_ttl` - Maximum time-to-live for the particle
pub fn random_particle<R: Rng, T>(
    rng: &mut R,
    width: f64,
    height: f64,
    max_ttl: usize,
    color_fn: impl FnOnce(&mut R) -> T,
) -> Particle<T> {
    let coords = Point2::new(rng.random_range(0.0..width), rng.random_range(0.0..height));
    let ttl = rng.random_range(0..max_ttl);
    let color = color_fn(rng);
    Particle { color, coords, ttl }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curl_noise_2d() {
        let coords = Point2::new(0.5, 0.5);
        let vector = curl_noise_2d(&coords, 1.0);
        // Should produce a finite vector
        assert!(vector.x.is_finite());
        assert!(vector.y.is_finite());
    }

    #[test]
    fn test_generate_vector_field() {
        let field = generate_vector_field(10, 10, 0.1, 1.0);
        assert_eq!(field.len(), 100);
        for vector in field {
            assert!(vector.x.is_finite());
            assert!(vector.y.is_finite());
        }
    }
}
