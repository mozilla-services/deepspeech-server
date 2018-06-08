extern crate futures;
extern crate hyper;
extern crate serde_json;

use args::TcpPort;

use self::futures::{future, Future, Stream};
use self::hyper::header::{HeaderValue, CONTENT_TYPE};
use self::hyper::service::service_fn;
use self::hyper::{Body, Method, Request, Response, Server, StatusCode};

use std::net::{IpAddr, SocketAddr};
use std::sync::mpsc::channel;
use std::sync::mpsc::Sender;

type ResponseFuture = Box<Future<Item = Response<Body>, Error = hyper::Error> + Send>;

use inference::InferenceResult;
use inference::RawAudioPCM;

static mut tx_audio: Option<Sender<(RawAudioPCM, Sender<InferenceResult>)>> = None;

fn http_handler(req: Request<Body>) -> ResponseFuture {
    debug!("Received HTTP: {} {}", req.method(), req.uri());
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/") => {
            debug!("POST connection accepted");
            let (parts, body) = req.into_parts();
            match parts.headers.get(CONTENT_TYPE) {
                Some(h) if h == HeaderValue::from_static("application/octet-stream") => {
                    debug!("This is valid: {:?}", h);
                    Box::new(body.concat2().map(|audio_content| {
                        let raw_pcm = audio_content.into_bytes();
                        debug!("RAW PCM is {:?} bytes", raw_pcm.len());
                        let inference_result = raw_pcm.len();
                        let infer = format!("inference: {}", inference_result);

                        let pcm = RawAudioPCM {
                            content: raw_pcm.clone(),
                        };

                        let (tx_string, rx_string) = channel();

                        unsafe {
                            match tx_audio {
                                Some(ref tx_audio_ok) => match tx_audio_ok
                                    .clone()
                                    .send((pcm, tx_string))
                                {
                                    Ok(_) => {
                                        debug!("Successfully sent message to thread");
                                        match rx_string.recv() {
                                            Ok(decoded_audio) => {
                                                info!("Received reply: {:?}", decoded_audio);
                                                Response::builder()
                                                    .status(StatusCode::OK)
                                                    .header(CONTENT_TYPE, "application/json")
                                                    .body(Body::from(
                                                        serde_json::to_string(&decoded_audio)
                                                            .unwrap(),
                                                    ))
                                                    .unwrap()
                                            }
                                            Err(err_recv) => {
                                                error!("Error trying to rx.recv(): {:?}", err_recv);
                                                Response::builder()
                                                    .status(StatusCode::NOT_FOUND)
                                                    .body(infer.into())
                                                    .unwrap()
                                            }
                                        }
                                    }
                                    Err(err) => {
                                        error!("Error while sending message to thread: {:?}", err);
                                        Response::builder()
                                            .status(StatusCode::NOT_FOUND)
                                            .body(infer.into())
                                            .unwrap()
                                    }
                                },
                                None => {
                                    error!("Unable to tx.send()");
                                    Response::builder()
                                        .status(StatusCode::NOT_FOUND)
                                        .body(infer.into())
                                        .unwrap()
                                }
                            }
                        }
                    }))
                }
                _ => Box::new(future::ok(
                    Response::builder()
                        .status(StatusCode::UNSUPPORTED_MEDIA_TYPE)
                        .body(Body::empty())
                        .unwrap(),
                )),
            }
        }
        _ => Box::new(future::ok(
            Response::builder()
                .status(StatusCode::METHOD_NOT_ALLOWED)
                .body(Body::empty())
                .unwrap(),
        )),
    }
}

pub fn th_http_listener(
    http_ip: IpAddr,
    http_port: TcpPort,
    _tx_audio: Sender<(RawAudioPCM, Sender<InferenceResult>)>,
) {
    unsafe {
        tx_audio = Some(_tx_audio);
    }

    let socket = SocketAddr::new(http_ip, http_port);
    info!("Building server http://{}", &socket);
    let server = Server::bind(&socket)
        .serve(|| service_fn(http_handler))
        .map_err(|e| eprintln!("server error: {}", e));
    info!("Listening on http://{}", socket);
    hyper::rt::run(server);
}
