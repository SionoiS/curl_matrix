use clap::{arg, App};
use curl_core::{generate_vector_field, Particle as CoreParticle};
use rand::prelude::*;
use rpi_led_matrix::{args, LedColor, LedMatrix};

/// Particle wrapper that adds the LED-specific color type.
struct Particle {
    inner: CoreParticle<LedColor>,
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
        )
        .arg(
            arg!(
            --"field-update-interval" <VAL> "'Frames between field recomputations'")
            .default_value("120")
            .required(false),
        )
        .arg(
            arg!(
            --"time-speed" <VAL> "'How fast time progresses in noise field'")
            .default_value("0.01")
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

    let field_update_interval: usize = matches
        .value_of_t("field-update-interval")
        .expect("Invalid value given for field-update-interval");

    let time_speed: f64 = matches
        .value_of_t("time-speed")
        .expect("Invalid value given for time-speed");

    println!("Options: {:?}", options);
    println!("Runtime Options: {:?}", rt_options);

    let matrix = LedMatrix::new(Some(options), Some(rt_options)).expect("Led Matrix Init");
    let mut canvas = matrix.offscreen_canvas();

    let (width, height) = canvas.canvas_size();
    let width = width as u32;
    let height = height as u32;

    println!(
        "Canvas {} x {} Particle count {} speed {} Field scale {} Flow scale {}",
        width, height, particle_count, particle_speed, field_scale, flow_speed
    );

    let mut current_time = 0.0;
    let mut vector_field = generate_vector_field(width, height, field_scale, current_time);

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

    let flow = nalgebra::vector![rng.random_range(0.0..1.0), rng.random_range(0.0..1.0)].normalize();

    let mut frame_count: usize = 0;

    loop {
        canvas.clear();

        for particle in particles.iter_mut() {
            // Quantize particle coordinates
            let x = particle.inner.coords.x.floor() as u32;
            let y = particle.inner.coords.y.floor() as u32;

            canvas.set(x as i32, y as i32, &particle.inner.color);

            // Get index from coords
            let idx = ((y.saturating_sub(1) * width) + x) as usize;
            let vector = vector_field[idx];

            // Move particle according to vector
            particle.inner.coords += (vector * particle_speed) + (flow * flow_speed);

            // Wrap around the edges
            if particle.inner.coords.x > width as f64 {
                particle.inner.coords.x = 0.0;
            }

            if particle.inner.coords.x < 0.0 {
                particle.inner.coords.x = width as f64;
            }

            if particle.inner.coords.y > height as f64 {
                particle.inner.coords.y = 0.0;
            }

            if particle.inner.coords.y < 0.0 {
                particle.inner.coords.y = height as f64;
            }

            particle.inner.ttl = particle.inner.ttl.saturating_sub(1);

            // Regenerate particle
            if particle.inner.ttl == 0 {
                *particle = random_particle(&mut rng, width as f64, height as f64, particle_ttl);
            }
        }

        canvas = matrix.swap(canvas);

        frame_count += 1;
        if frame_count.is_multiple_of(field_update_interval) {
            current_time += time_speed;
            vector_field = generate_vector_field(width, height, field_scale, current_time);
        }
    }
}

fn random_particle<R: Rng>(rng: &mut R, width: f64, height: f64, max_ttl: usize) -> Particle {
    Particle {
        inner: curl_core::random_particle(rng, width, height, max_ttl, |rng| {
            let mut rgb = [0u8; 3];
            rng.fill_bytes(&mut rgb);
            LedColor {
                red: rgb[0],
                green: rgb[1],
                blue: rgb[2],
            }
        }),
    }
}
