use fastcwt::*;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use hound::WavReader;
use num::Rational64;
use scop::Defs;
use std::sync::{Arc, Mutex};
use weresocool_ast::{NormalForm, PointOp};
use weresocool_core::{manager::RenderManager, portaudio::real_time_render_manager};
use weresocool_error::Error;
use weresocool_instrument::renderable::{nf_to_vec_renderable, renderables_to_render_voices};
use weresocool_instrument::Basis;

const SAMPLE_RATE: usize = 11025;

fn main() {
    let mut reader = WavReader::open("simple.wav").unwrap();
    // let duration_samples = reader.duration();

    let mut input = vec![];
    for sample in reader.samples::<f32>() {
        input.push(sample.unwrap());
    }

    let resampled_input = resample(
        &[input],
        reader.spec().sample_rate as f64,
        SAMPLE_RATE as f64,
        2.0,
        256,
        256,
        // reader.spec().channels as usize,
    )
    .unwrap();

    let cwt_output = perform_cwt(&resampled_input[0]);
    dbg!("finished cwt");

    let mut operations: Vec<Vec<PointOp>> = Vec::new();
    let window_size = 1024;

    for time_slice in cwt_output {
        let peaks = identify_peaks(&time_slice, SAMPLE_RATE, 10);
        let mut ops: Vec<PointOp> = Vec::new();

        for peak in peaks {
            if peak.freq.numer() > &(20 as i64) && peak.magnitude > 0.0 {
                let point_op = PointOp {
                    fm: Rational64::new(*peak.freq.numer() as i64, *peak.freq.denom() as i64),
                    fa: Rational64::new(0, 1),
                    pm: Rational64::new(0, 1),
                    pa: Rational64::new(0, 1),
                    g: Rational64::new((peak.magnitude * 1000.0).round() as i64, 1000),
                    l: Rational64::new(window_size as i64, SAMPLE_RATE as i64),
                    ..PointOp::default()
                };

                ops.push(point_op);
            }
        }

        if !ops.is_empty() {
            operations.push(ops);
        }
    }

    dbg!(operations.len());
    dbg!(operations[0].len());
    dbg!(operations[0].clone());

    let normal_form = NormalForm {
        operations,
        length_ratio: Rational64::new(1, 1),
    };

    play(normal_form).unwrap();
}

fn perform_cwt(input: &[f32]) -> Vec<Vec<num::Complex<f64>>> {
    let window_size = 1024;
    let wavelet = Wavelet::create(1.0);
    let mut transform = FastCWT::create(wavelet, true);

    let mut result = Vec::new();
    let mut window_vec = Vec::with_capacity(window_size); // Preallocate vector
    for window in input.windows(window_size) {
        window_vec.clear();
        window_vec.extend(window.iter().map(|&x| x as f64)); // Reuse the same vector for each window
        let scale = Scales::create(ScaleTypes::LinFreq, 48000, 20.0, 20000.0, 1000);
        let window_result = transform.cwt(window_size, &window_vec, scale);
        result.push(window_result);
    }

    result
}

#[derive(Debug)]
pub struct Peak {
    pub freq: Rational64,
    pub magnitude: f32,
}

pub fn identify_peaks(
    data: &[num::Complex<f64>],
    sample_rate: usize,
    num_peaks: usize,
) -> Vec<Peak> {
    let mut peaks = Vec::new();

    let magnitudes: Vec<f64> = data.iter().map(|c| c.norm()).collect();

    let max_magnitude = magnitudes.iter().cloned().fold(0. / 0., f64::max); // Find maximum magnitude

    let n = magnitudes.len();

    for i in 1..(n - 1) {
        if magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1] {
            let freq = Rational64::new((i * 220) as i64, n as i64).reduced(); // frequency as ratio from fundamental
            peaks.push(Peak {
                freq,
                magnitude: (magnitudes[i] / max_magnitude) as f32, // Normalize magnitude
            });
        }
    }

    // Sort the peaks by magnitude in descending order
    peaks.sort_by(|a, b| b.magnitude.partial_cmp(&a.magnitude).unwrap());

    // Return only the top N peaks
    peaks.truncate(num_peaks);

    peaks
}

fn play(nf: NormalForm) -> Result<(), Error> {
    dbg!("playing");
    weresocool_shared::Settings::init_default();
    let basis = Basis {
        f: Rational64::new(220, 1),
        g: Rational64::new(1, 1),
        l: Rational64::new(1, 1),
        p: Rational64::new(0, 1),
        a: Rational64::new(1, 1),
        d: Rational64::new(1, 1),
    };

    let (tx, rx) = std::sync::mpsc::channel::<bool>();
    let mut table = Defs::new();
    let renderables = nf_to_vec_renderable(&nf, &mut table, &basis)?;
    let render_voices = renderables_to_render_voices(renderables);

    let render_manager = Arc::new(Mutex::new(RenderManager::init(None, Some(tx), true, None)));

    let mut stream = real_time_render_manager(Arc::clone(&render_manager))?;

    stream.start()?;
    render_manager
        .lock()
        .unwrap()
        .push_render(render_voices, true);
    dbg!("pushed");
    match rx.recv() {
        Ok(_) => {}
        Err(e) => {
            println!("{}", e);
            std::process::exit(1);
        }
    };
    stream.stop()?;

    Ok(())
}

fn resample(
    input_data: &[Vec<f32>],
    from_sample_rate: f64,
    to_sample_rate: f64,
    transition_bandwidth: f64,
    sinc_len: usize,
    oversampling_factor: usize,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    // Define the resampler parameters
    let params = SincInterpolationParameters {
        sinc_len,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor,
        window: WindowFunction::BlackmanHarris2,
    };

    // Create a resampler
    let mut resampler = SincFixedIn::<f32>::new(
        to_sample_rate / from_sample_rate,
        transition_bandwidth,
        params,
        input_data[0].len(),
        1, // channels,
    )?;

    // Resample the data
    let output_data = resampler.process(input_data, None)?;

    Ok(output_data)
}
