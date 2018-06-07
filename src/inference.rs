extern crate serde;

extern crate deepspeech;
extern crate audrey;
extern crate futures;

extern crate bytes;

use self::deepspeech::Model;
use self::audrey::Format;
use self::audrey::read::Reader;
use self::audrey::read::Description;
use self::bytes::Bytes;

use std::sync::mpsc::{Receiver, SyncSender};
use std::vec::Vec;
use std::io::Cursor;
use std::path::Path;

use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct RawAudioPCM {
	pub content: Bytes
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceData {
	text: String,
	confidence: f32
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResult {
	status: String,
	data: Vec<InferenceData>
}

const N_CEP :u16 = 26;
const N_CONTEXT :u16 = 9;
const BEAM_WIDTH :u16 = 500;

const LM_WEIGHT :f32 = 1.75;
const WORD_COUNT_WEIGHT :f32 = 1.0;
const VALID_WORD_COUNT_WEIGHT :f32 = 1.0;

// The model has been trained on this specific
// sample rate.
const AUDIO_SAMPLE_RATE :u32 = 16000;
const AUDIO_CHANNELS :u32	= 1;
const AUDIO_FORMAT :Format   = Format::Wav;

fn start_model(model: String, alphabet: String, lm: String, trie: String)
	-> Model
{
	let mut m = Model::load_from_files(
		Path::new(&model),
		N_CEP,
		N_CONTEXT,
		Path::new(&alphabet),
		BEAM_WIDTH);

	m.enable_decoder_with_lm(
		Path::new(&alphabet),
		Path::new(&lm),
		Path::new(&trie),
		LM_WEIGHT,
		WORD_COUNT_WEIGHT,
		VALID_WORD_COUNT_WEIGHT);

	m
}

fn ensure_valid_audio(desc: Description) -> bool
{
	if desc.format() != AUDIO_FORMAT {
		error!("Invalid audio format: {:?}", desc.format());
		return false;
	};

	if desc.channel_count() != AUDIO_CHANNELS {
		error!("Invalid number of channels: {}", desc.channel_count());
		return false;
	};

	if desc.sample_rate() != AUDIO_SAMPLE_RATE {
		error!("Invalid sample rate: {}", desc.sample_rate());
		return false;
	}

	true
}

fn inference_result(result: String, status: bool)
	-> InferenceResult
{
	let confidence_value = match status {
		true  => 1.0,
		false => 0.0
	};

	let status_value = match status {
		true  => "ok".to_string(),
		false => "ko".to_string()
	};

	let mut inf_data: Vec<InferenceData> = Vec::new();
	inf_data.push(InferenceData {
		confidence: confidence_value,
		text: result
	});

	let inf_result = InferenceResult {
		status: status_value,
		data: inf_data
	};

	inf_result
}

pub fn th_inference(model: String, alphabet: String, lm: String, trie: String, rx_audio: Receiver<RawAudioPCM>, tx_string: SyncSender<InferenceResult>) {
	info!("Inference thread started");
	let mut model = start_model(model, alphabet, lm, trie);

	loop {
		info!("Waiting ...");
		let inf_result = match rx_audio.recv() {
			Ok(audio) => {
				info!("Received message: {:?} bytes", audio.content.len());

				let inf = match Reader::new(Cursor::new(&*audio.content)) {
					Ok(mut reader) => {
						let desc = reader.description();

						match ensure_valid_audio(desc) {
							true  => {
								let audio_buf :Vec<_> = reader.samples().map(|s| s.unwrap()).collect::<Vec<_>>();
								let start  = Instant::now();
								let result = model.speech_to_text(&audio_buf, AUDIO_SAMPLE_RATE).unwrap();
								let duration = start.elapsed();
								info!("Inference took: {:?}", duration);
								inference_result(result, true)
							},
							false => inference_result("".to_string(), false)
						}
					},

					Err(err)   => {
						error!("Audrey read error: {:?}", err);
						inference_result("".to_string(), false)
					}
				};

				inf
			},

			Err(err_recv)  => {
				error!("Error trying to rx.recv(): {:?}", err_recv);
				inference_result("".to_string(), false)
			}
		};

		tx_string.send(inf_result);
	}
	
}
