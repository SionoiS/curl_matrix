use clap::App;

use rpi_led_matrix::{args, LedMatrix};

use embedded_graphics::{
    image::{Image, ImageRawBE},
    pixelcolor::Rgb888,
    prelude::*,
    Drawable,
};

const IMAGE_DATA: &[u8] = include_bytes!("../assets/ferris_test_card.rgb");
const IMAGE_SIZE: i32 = 64;

fn main() {
    let app = args::add_matrix_args(App::new("Rust Test Image"));
    let matches = app.get_matches();
    let (options, rt_options) = args::matrix_options_from_args(&matches);

    println!("Options: {:?}", options);
    println!("Runtime Options: {:?}", rt_options);

    let matrix = LedMatrix::new(Some(options), Some(rt_options)).expect("Led Matrix Init");
    let mut canvas = matrix.offscreen_canvas();

    let (width, height) = canvas.canvas_size();

    println!("Canvas: {}x{}", width, height);

    let image_data = ImageRawBE::<Rgb888>::new(IMAGE_DATA, IMAGE_SIZE as u32);
    let image = Image::new(
        &image_data,
        Point::new(
            (height / 2 - IMAGE_SIZE / 2) as i32,
            (width / 2 - IMAGE_SIZE / 2) as i32,
        ),
    );

    loop {
        canvas.clear();

        image.draw(&mut canvas).unwrap();

        canvas = matrix.swap(canvas);
    }
}
