mod simplex;

use std::io::Write;

use nalgebra::{Point2, Point3, Vector2};

use rpi_led_panel::{RGBMatrix, RGBMatrixConfig};

use rand::prelude::*;

struct Particle {
    r: u8,
    g: u8,
    b: u8,
    coords: Point2<f64>,
}

const PARTICLE_COUNT: usize = 256;
const PARTICLE_SPEED: f64 = 0.1;

fn main() {
    let config: RGBMatrixConfig = argh::from_env();

    let (mut matrix, mut canvas) = RGBMatrix::new(config, 0).expect("Led Matrix Init");

    let rows = canvas.rows(); // 64
    let columns = canvas.cols(); // 64
    let pixel_count = rows * columns; // 4096

    let mut vector_field = Vec::with_capacity(pixel_count);

    // Vector field init
    for y in 0..columns {
        for x in 0..rows {
            let coords = Point2::new(x as f64, y as f64);

            let vector = curl_noise_2d(&coords, 1f64);

            vector_field.push(vector);
        }
    }

    // Init particles with random colors and position
    let mut particles = Vec::with_capacity(PARTICLE_COUNT);
    let mut rng = rand::thread_rng();
    for _ in 0..PARTICLE_COUNT {
        let mut rgb = [0u8; 3];
        rng.fill_bytes(&mut rgb);

        let x = rng.gen_range(0.0..64.0);
        let y = rng.gen_range(0.0..64.0);

        let part = Particle {
            r: rgb[0],
            g: rgb[1],
            b: rgb[2],
            coords: Point2::new(x, y),
        };

        particles.push(part);
    }

    for step in 0.. {
        canvas.fill(0, 0, 0);

        for particle in particles.iter_mut() {
            // Quantize particle coordinates
            let x = particle.coords.x.floor() as usize;
            let y = particle.coords.y.floor() as usize;

            canvas.set_pixel(x, y, particle.r, particle.g, particle.b);

            // Get index from coords
            let idx = y.saturating_sub(1) * rows + x;
            let vector = vector_field[idx];

            // Move particle according to vector
            particle.coords += vector * PARTICLE_SPEED;

            // Wrap around the edges
            if particle.coords.x > 64.0 {
                particle.coords.x = 0.0;
            }

            if particle.coords.x < 0.0 {
                particle.coords.x = 64.0;
            }

            if particle.coords.y > 64.0 {
                particle.coords.y = 0.0;
            }

            if particle.coords.y < 0.0 {
                particle.coords.y = 64.0;
            }
        }

        canvas = matrix.update_on_vsync(canvas);

        if step % 120 == 0 {
            print!("\r{:>100}\rFramerate: {}", "", matrix.get_framerate());
            std::io::stdout().flush().unwrap();
        }
    }
}

pub fn curl_noise_2d(coordinates: &Point2<f64>, time: f64) -> Vector2<f64> {
    let space_time = Point3::new(coordinates.x, coordinates.y, time);

    let (_, deriv) = simplex::with_derivatives_3d(&space_time);

    let derivatives = &Vector2::new(deriv.x, deriv.y);

    curl_2d(derivatives)
}

fn curl_2d(derivatives: &Vector2<f64>) -> Vector2<f64> {
    // potential field deriv y -> vector field x
    // potential field deriv -x -> vector field y
    Vector2::new(derivatives.y, -derivatives.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    use embedded_graphics::{
        image::{Image, ImageRawBE},
        pixelcolor::Rgb888,
        prelude::*,
        Drawable,
    };

    const IMAGE_DATA: &[u8] = include_bytes!("../assets/ferris_test_card.rgb");
    const IMAGE_SIZE: usize = 64;

    #[test]
    fn test_image() {
        let config: RGBMatrixConfig = argh::from_env();

        let rows = config.rows;
        let cols = config.cols;

        let (mut matrix, mut canvas) =
            RGBMatrix::new(config, 0).expect("Matrix initialization failed");

        let image_data = ImageRawBE::<Rgb888>::new(IMAGE_DATA, IMAGE_SIZE as u32);
        let image = Image::new(
            &image_data,
            Point::new(
                (cols / 2 - IMAGE_SIZE / 2) as i32,
                (rows / 2 - IMAGE_SIZE / 2) as i32,
            ),
        );

        for step in 0.. {
            canvas.fill(0, 0, 0);
            image.draw(canvas.as_mut()).unwrap();
            canvas = matrix.update_on_vsync(canvas);

            if step % 120 == 0 {
                print!("\r{:>100}\rFramerate: {}", "", matrix.get_framerate());
                std::io::stdout().flush().unwrap();
            }
        }
    }
}
