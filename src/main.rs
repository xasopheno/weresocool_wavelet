use fastcwt::*;
use hound::WavReader;
use num::Rational64;
use std::fs::File;
use weresocool_ast::{NormalForm, PointOp};

fn main() {
    let mut reader = WavReader::open("simple.wav").unwrap();
    let duration_samples = reader.duration();

    let mut input = vec![];
    for sample in reader.samples::<f32>() {
        input.push(sample.unwrap());
    }

    let cwt_output = perform_cwt(&input);
    dbg!("finished cwt");

    let mut operations: Vec<Vec<PointOp>> = Vec::new();
    let window_size = 1024;

    for time_slice in cwt_output {
        let peaks = identify_peaks(&time_slice, reader.spec().sample_rate as usize);
        let mut ops: Vec<PointOp> = Vec::new();

        for peak in peaks {
            let point_op = PointOp {
                fm: Rational64::new(*peak.freq.numer() as i64, *peak.freq.denom() as i64), // frequency as ratio from fundamental
                fa: Rational64::new(0, 1),
                pm: Rational64::new(0, 1),
                pa: Rational64::new(0, 1),
                g: Rational64::new((peak.magnitude * 1000.0).round() as i64, 1000), // gain from CWT peak magnitude
                l: Rational64::new(window_size as i64, reader.spec().sample_rate as i64), // length based on window size and sample rate
                ..PointOp::default()
            };

            ops.push(point_op);
        }

        operations.push(ops);
    }
    dbg!(operations);
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

pub fn identify_peaks(data: &[num::Complex<f64>], sample_rate: usize) -> Vec<Peak> {
    let mut peaks = Vec::new();

    let magnitudes: Vec<f64> = data.iter().map(|c| c.norm()).collect();

    let n = magnitudes.len();

    for i in 1..(n - 1) {
        if magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1] {
            let freq = Rational64::new((i * sample_rate) as i64, n as i64).reduced(); // frequency as ratio from fundamental
            peaks.push(Peak {
                freq,
                magnitude: magnitudes[i] as f32,
            });
        }
    }

    peaks
}
