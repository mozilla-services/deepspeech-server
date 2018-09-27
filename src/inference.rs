extern crate serde;

extern crate audrey;
extern crate deepspeech;
extern crate futures;

extern crate mkstemp;

extern crate byte_slice_cast;
extern crate bytes;

use self::audrey::read::Description;
use self::audrey::read::Reader;
use self::audrey::Format;
use self::byte_slice_cast::*;
use self::bytes::Bytes;
use self::deepspeech::Model;

use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;
use std::vec::Vec;

#[derive(Debug)]
pub struct RawAudioPCM {
    pub content: Bytes,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceData {
    text: String,
    confidence: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResult {
    status: String,
    data: Vec<InferenceData>,
}

const N_CEP: u16 = 26;
const N_CONTEXT: u16 = 9;
const BEAM_WIDTH: u16 = 500;

const LM_WEIGHT: f32 = 1.75;
const VALID_WORD_COUNT_WEIGHT: f32 = 1.0;

// The model has been trained on this specific
// sample rate.
const AUDIO_SAMPLE_RATE: u32 = 16000;
const AUDIO_CHANNELS: u32 = 1;
const AUDIO_FORMAT: Format = Format::Wav;

fn start_model(model: String, alphabet: String, lm: String, trie: String) -> Model {
    let mut m = Model::load_from_files(
        Path::new(&model),
        N_CEP,
        N_CONTEXT,
        Path::new(&alphabet),
        BEAM_WIDTH,
    ).unwrap();

    m.enable_decoder_with_lm(
        Path::new(&alphabet),
        Path::new(&lm),
        Path::new(&trie),
        LM_WEIGHT,
        VALID_WORD_COUNT_WEIGHT,
    );

    m
}

fn ensure_valid_audio(desc: Description) -> bool {
    let rv_format = if desc.format() != AUDIO_FORMAT {
        error!("Invalid audio format: {:?}", desc.format());
        false
    } else {
        true
    };

    let rv_channels = if desc.channel_count() != AUDIO_CHANNELS {
        error!("Invalid number of channels: {}", desc.channel_count());
        false
    } else {
        true
    };

    let rv_rate = if desc.sample_rate() != AUDIO_SAMPLE_RATE {
        error!("Invalid sample rate: {}", desc.sample_rate());
        false
    } else {
        true
    };

    rv_format && rv_channels && rv_rate
}

fn inference_result(result: String, status: bool) -> InferenceResult {
    let confidence_value = match status {
        true => 1.0,
        false => 0.0,
    };

    let status_value = match status {
        true => "ok".to_string(),
        false => "ko".to_string(),
    };

    let mut inf_data: Vec<InferenceData> = Vec::new();
    inf_data.push(InferenceData {
        confidence: confidence_value,
        text: result,
    });

    let inf_result = InferenceResult {
        status: status_value,
        data: inf_data,
    };

    inf_result
}

fn inference_error() -> InferenceResult {
    inference_result("".to_string(), false)
}

fn inference(m: &mut Model, buffer: &[i16]) -> InferenceResult {
    let start = Instant::now();

    let rv = match m.speech_to_text(buffer, AUDIO_SAMPLE_RATE) {
        Ok(result) => inference_result(result, true),
        Err(err) => {
            error!("Error while running inference: {:?}", err);
            inference_error()
        }
    };

    let duration = start.elapsed();
    info!("Inference took: {:?}", duration);

    rv
}

fn maybe_dump_debug(stream: Bytes, directory: String) {
    use self::mkstemp::TempFile;
    use std::io::Write;

    let temp_root = Path::new(&directory);
    let temp_file_name = temp_root.join("ds-debug-wav-XXXXXX");

    debug!(
        "Dumping RAW PCM content to {:?} => {:?}",
        temp_root, temp_file_name
    );

    match TempFile::new(temp_file_name.to_str().unwrap(), false) {
        Ok(mut file) => match file.write(&*stream) {
            Ok(_) => debug!("Sucessfully write debug file"),
            Err(err) => error!("Error writing content of debug file {:?}", err),
        },
        Err(err) => error!("Error creating debug file: {:?}", err),
    }
}

fn maybe_warmup_model(mut m: &mut Model, directory: String, cycles: i32) {
    let warmup_dir = Path::new(&directory);
    let mut allwaves = Vec::new();

    for entry in warmup_dir.read_dir().expect("read_dir call failed") {
        if let Ok(entry) = entry {
            match entry.path().extension() {
                Some(ext) if ext == "wav" => {
                    debug!("Found one more WAV file: {:?}", entry.path());
                    allwaves.push(entry.path());
                }
                Some(_) => {}
                None => {}
            }
        }
    }

    for wave in allwaves.iter() {
        debug!("Warmup with {:?}", wave);
        if let Ok(audio_file) = File::open(wave) {
            if let Ok(mut reader) = Reader::new(audio_file) {
                let audio_buf: Vec<_> = reader.samples().map(|s| s.unwrap()).collect::<Vec<_>>();
                for i in 0..cycles {
                    info!("Warmup cycle {} of {}", i + 1, cycles);
                    inference(&mut m, &*audio_buf);
                }
            }
        }
    }
}

pub fn th_inference(
    model: String,
    alphabet: String,
    lm: String,
    trie: String,
    rx_audio: Receiver<(RawAudioPCM, Sender<InferenceResult>)>,
    dump_dir: String,
    warmup_dir: String,
    warmup_cycles: i32,
) {
    info!("Inference thread started");
    let mut model_instance = start_model(model, alphabet, lm, trie);

    if warmup_dir.len() > 0 {
        maybe_warmup_model(&mut model_instance, warmup_dir.clone(), warmup_cycles);
    }

    loop {
        info!("Model ready and waiting for data to infer ...");
        match rx_audio.recv() {
            Ok((audio, tx_string)) => {
                info!("Received message: {:?} bytes", audio.content.len());

                #[cfg(feature = "dump_debug_stream")]
                maybe_dump_debug(audio.content.clone(), dump_dir.clone());

                let inf = match Reader::new(Cursor::new(&*audio.content)) {
                    Ok(mut reader) => {
                        let desc = reader.description();

                        match ensure_valid_audio(desc) {
                            true => {
                                let audio_buf: Vec<_> =
                                    reader.samples().map(|s| s.unwrap()).collect::<Vec<_>>();
                                inference(&mut model_instance, &*audio_buf)
                            }

                            false => inference_error(),
                        }
                    }

                    Err(err) => {
                        error!("Audrey read error: {:?}", err);
                        let mut audio_u8 = audio.content.to_vec();
                        match audio_u8.as_mut_slice_of::<i16>() {
                            Ok(audio_i16) => {
                                info!("Trying with RAW PCM {:?} bytes", audio_i16.len());
                                inference(&mut model_instance, &*audio_i16)
                            }
                            Err(err) => {
                                error!("Unable to make u8 -> i16: {:?}", err);
                                inference_error()
                            }
                        }
                    }
                };

                match tx_string.send(inf) {
                    Ok(_) => {}
                    Err(err) => error!("Error sending inference result: {:?}", err),
                }
            }

            Err(err_recv) => error!("Error trying to rx.recv(): {:?}", err_recv),
        }
    }
}
