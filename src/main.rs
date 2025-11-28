mod simplex;

use clap::{arg, App};

use rpi_led_matrix::{args, LedColor, LedMatrix};

use nalgebra::{vector, Point2, Point3, Vector2};

use rand::prelude::*;

struct Particle {
    color: LedColor,
    coords: Point2<f64>,
    ttl: usize,
}

fn main() {
    let app = App::new("Curl Noise Led Matrix")
        .arg(
            arg!(
            --"particle-count" <VAL> "'Total number of particles'")
            .default_value("2000")
            .required(false),
        )
        .arg(
            arg!(
            --"field-scale" <VAL> "'Scaling of the vector field'")
            .default_value("0.05")
            .required(false),
        )
        .arg(
            arg!(
            --"particle-speed" <VAL> "'Scaling of the particle movements'")
            .default_value("0.01")
            .required(false),
        )
        .arg(
            arg!(
            --"flow-speed" <VAL> "'Scaling of the flow'")
            .default_value("0.09")
            .required(false),
        )
        .arg(
            arg!(
            --"particle-ttl" <VAL> "'Max particle life time'")
            .default_value("2500")
            .required(false),
        );

    let app = args::add_matrix_args(app);
    let matches = app.get_matches();
    let (options, rt_options) = args::matrix_options_from_args(&matches);

    let particle_count: usize = matches
        .value_of_t("particle-count")
        .expect("Invalid value given for particle-count");

    let field_scale: f64 = matches
        .value_of_t("field-scale")
        .expect("Invalid value given for field-scale");

    let particle_speed: f64 = matches
        .value_of_t("particle-speed")
        .expect("Invalid value given for particle-speed");

    let flow_speed: f64 = matches
        .value_of_t("flow-speed")
        .expect("Invalid value given for flow-speed");

    let particle_ttl: usize = matches
        .value_of_t("particle-ttl")
        .expect("Invalid value given for particle-ttl");

    println!("Options: {:?}", options);
    println!("Runtime Options: {:?}", rt_options);

    let matrix = LedMatrix::new(Some(options), Some(rt_options)).expect("Led Matrix Init");
    let mut canvas = matrix.offscreen_canvas();

    let (width, height) = canvas.canvas_size();
    let width = width as u32;
    let height = height as u32;
    let pixel_count = width * height; // 4096

    println!(
        "Canvas {} x {} Particle count {} speed {} Field scale {} Flow scale {}",
        width, height, particle_count, particle_speed, field_scale, flow_speed
    );

    let mut vector_field = Vec::with_capacity(pixel_count as usize);

    // Vector field init
    for y in 0..height {
        for x in 0..width {
            let coords = Point2::new(x as f64 * field_scale, y as f64 * field_scale);

            let vector = curl_noise_2d(&coords, 1f64); //TODO add time dimension

            vector_field.push(vector);
        }
    }

    // Init particles with random colors and position
    let mut particles = Vec::with_capacity(particle_count);
    let mut rng = rand::rng();
    for _ in 0..particle_count {
        particles.push(random_particle(
            &mut rng,
            width as f64,
            height as f64,
            particle_ttl,
        ));
    }

    let flow = vector![rng.random_range(0.0..1.0), rng.random_range(0.0..1.0)].normalize();

    loop {
        canvas.clear();

        for particle in particles.iter_mut() {
            // Quantize particle coordinates
            let x = particle.coords.x.floor() as u32;
            let y = particle.coords.y.floor() as u32;

            canvas.set(x as i32, y as i32, &particle.color);

            // Get index from coords
            let idx = ((y.saturating_sub(1) * width) + x) as usize;
            let vector = vector_field[idx];

            // Move particle according to vector
            particle.coords += (vector * particle_speed) + (flow * flow_speed);

            // Wrap around the edges
            if particle.coords.x > width as f64 {
                particle.coords.x = 0.0;
            }

            if particle.coords.x < 0.0 {
                particle.coords.x = width as f64;
            }

            if particle.coords.y > height as f64 {
                particle.coords.y = 0.0;
            }

            if particle.coords.y < 0.0 {
                particle.coords.y = height as f64;
            }

            particle.ttl = particle.ttl.saturating_sub(1);

            //Regenerate particle
            if particle.ttl <= 0 {
                *particle = random_particle(&mut rng, width as f64, height as f64, particle_ttl);
            }
        }

        canvas = matrix.swap(canvas);
    }
}

fn random_particle<R: Rng>(rng: &mut R, width: f64, height: f64, max_ttl: usize) -> Particle {
    let mut rgb = [0u8; 3];
    rng.fill_bytes(&mut rgb);

    let color = LedColor {
        red: rgb[0],
        green: rgb[1],
        blue: rgb[2],
    };
    let coords = Point2::new(rng.random_range(0.0..width), rng.random_range(0.0..height));
    let ttl = rng.random_range(0..max_ttl);

    return Particle { color, coords, ttl };
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
