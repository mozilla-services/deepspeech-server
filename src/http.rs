extern crate hyper;
extern crate futures;
extern crate serde_json;

use args::TcpPort;

use self::futures::{future, Future, Stream};
use self::hyper::{Body, Method, Request, Response, Server, StatusCode};
use self::hyper::service::service_fn;
use self::hyper::header::{CONTENT_TYPE, HeaderValue};

use std::sync::mpsc::{Receiver, SyncSender};
use std::net::{IpAddr, SocketAddr};

type ResponseFuture = Box<Future<Item=Response<Body>, Error=hyper::Error> + Send>;

use inference::RawAudioPCM;
use inference::InferenceResult;

static mut tx_audio: Option<SyncSender<RawAudioPCM>> = None;
static mut rx_string: Option<Receiver<InferenceResult>> = None;

fn http_handler(req: Request<Body>) -> ResponseFuture {
	debug!("Received HTTP: {} {}", req.method(), req.uri());
	match (req.method(), req.uri().path()) {
		(&Method::POST, "/") => {
			debug!("POST connection accepted");
			let (parts, body) = req.into_parts();
			match parts.headers.get(CONTENT_TYPE) {
				Some(h) if h == HeaderValue::from_static("application/octet-stream") => {
					debug!("This is valid: {:?}", h);
					Box::new(
						body.concat2().map(|audio_content| {
							let raw_pcm = audio_content.into_bytes();
							debug!("RAW PCM is {:?} bytes", raw_pcm.len());
							let inference_result = raw_pcm.len();
							let infer = format!("inference: {}", inference_result);

							let pcm = RawAudioPCM {
								content: raw_pcm.clone()
							};

							unsafe {
								match tx_audio.as_ref().unwrap().clone().send(pcm) {
									Ok(_)	=> debug!("Successfully sent message to thread"),
									Err(err) => error!("Error while sending message to thread: {:?}", err)
								}
							}

							unsafe {
								match rx_string.as_ref().unwrap().clone().recv() {
									Ok(decoded_audio) => {
										info!("Received reply: {:?}", decoded_audio);
										Response::builder()
												.status(StatusCode::OK)
												.header(CONTENT_TYPE, "application/json")
												.body(Body::from(serde_json::to_string(&decoded_audio).unwrap()))
												.unwrap()
									},
									Err(err_recv)  => {
										error!("Error trying to rx.recv(): {:?}", err_recv);
										Response::builder()
												.status(StatusCode::NOT_FOUND)
												.body(infer.into())
												.unwrap()
									}
								}
							}
						})
					)
				},
				_ => {
					Box::new(future::ok(Response::builder()
										.status(StatusCode::UNSUPPORTED_MEDIA_TYPE)
										.body(Body::empty())
										.unwrap()))
				}
			}
		},
		_ => {
			Box::new(future::ok(Response::builder()
								.status(StatusCode::METHOD_NOT_ALLOWED)
								.body(Body::empty())
								.unwrap()))
		}
	}
}

pub fn th_http_listener(http_ip: IpAddr, http_port: TcpPort, _tx_audio: SyncSender<RawAudioPCM>, _rx_string: Receiver<InferenceResult>) {
	unsafe {
		tx_audio  = Some(_tx_audio);
		rx_string = Some(_rx_string);

	}

	let socket = SocketAddr::new(http_ip, http_port);
	info!("Building server http://{}", &socket);
	let server = Server::bind(&socket)
		.serve(|| service_fn(http_handler))
		.map_err(|e| eprintln!("server error: {}", e));
	info!("Listening on http://{}", socket);
	hyper::rt::run(server);
}
