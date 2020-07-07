#[macro_use]
extern crate log;
#[macro_use]
extern crate serde_derive;

extern crate simplelog;

use std::sync::mpsc::channel;
use std::thread;

mod args;
use args::ArgsParser;

mod http;
use http::th_http_listener;

mod inference;
use inference::th_inference;

fn main() {
    let rc = ArgsParser::from_cli();

    let log_level = rc.verbosity_level.into();
    let _ = simplelog::TermLogger::init(log_level, simplelog::Config::default());

    debug!("Parsed all CLI args: {:?}", rc);

    let (tx_audio, rx_audio) = channel();

    let mut threads = Vec::new();
    let rc_inference = rc.clone();
    let thread_inference = thread::Builder::new()
        .name("InferenceService".to_string())
        .spawn(move || {
            th_inference(
                rc_inference.model,
                rc_inference.scorer,
                rx_audio,
                rc_inference.dump_dir,
                rc_inference.warmup_dir,
                rc_inference.warmup_cycles,
            );
        });
    threads.push(thread_inference);

    let rc_http = rc.clone();
    let thread_http = thread::Builder::new()
        .name("HttpService".to_string())
        .spawn(move || {
            th_http_listener(rc_http.http_ip, rc_http.http_port, tx_audio);
        });
    threads.push(thread_http);

    println!("Started all thread.");

    for hdl in threads {
        if hdl.is_ok() {
            hdl.unwrap().join().unwrap();
        }
    }
}
