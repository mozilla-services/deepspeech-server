extern crate clap;
extern crate simplelog;

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::str::FromStr;

pub type TcpPort = u16;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum VerbosityLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
}

impl Into<simplelog::LevelFilter> for VerbosityLevel {
    fn into(self) -> simplelog::LevelFilter {
        match self {
            VerbosityLevel::DEBUG => simplelog::LevelFilter::Debug,
            VerbosityLevel::INFO => simplelog::LevelFilter::Info,
            VerbosityLevel::WARN => simplelog::LevelFilter::Warn,
            VerbosityLevel::ERROR => simplelog::LevelFilter::Error,
        }
    }
}

#[derive(Debug, Clone)]
/// Holds the program's runtime configuration
pub struct RuntimeConfig {
    pub http_ip: IpAddr,
    pub http_port: TcpPort,
    pub dump_dir: String,
    pub model: String,
    pub alphabet: String,
    pub lm: String,
    pub trie: String,
    pub verbosity_level: VerbosityLevel,
}

pub struct ArgsParser;

impl ArgsParser {
    fn to_ip_addr(o: Option<&str>) -> IpAddr {
        let default_ip = IpAddr::V6(Ipv6Addr::from_str("::0").unwrap());
        match o {
            Some(ip_str) => {
                if Ipv6Addr::from_str(ip_str).is_ok() {
                    IpAddr::V6(Ipv6Addr::from_str(ip_str).unwrap())
                } else if Ipv4Addr::from_str(ip_str).is_ok() {
                    IpAddr::V4(Ipv4Addr::from_str(ip_str).unwrap())
                } else {
                    default_ip
                }
            }
            None => default_ip,
        }
    }

    fn to_port(o: Option<&str>) -> TcpPort {
        let default_port = 8080;
        match o.unwrap_or(default_port.to_string().as_str())
            .parse::<u16>()
        {
            Ok(rv) => rv,
            Err(_) => default_port,
        }
    }

    fn to_verbosity_level(occ: u64) -> VerbosityLevel {
        match occ {
            0 => VerbosityLevel::ERROR,
            1 => VerbosityLevel::WARN,
            2 => VerbosityLevel::INFO,
            3 => VerbosityLevel::DEBUG,
            _ => VerbosityLevel::DEBUG,
        }
    }

    pub fn from_cli() -> RuntimeConfig {
        let matches = clap::App::new("DeepSpeech Inference Server")
            .version("0.1")
            .author("<lissyx@lissyx.dyndns.org>")
            .about("Running inference from POST-ed RAW PCM.")
            .arg(
                clap::Arg::with_name("http_ip")
                    .short("h")
                    .long("http_ip")
                    .value_name("HTTP_IP")
                    .help("IP address to listen on for HTTP")
                    .takes_value(true)
                    .required(false),
            )
            .arg(
                clap::Arg::with_name("http_port")
                    .short("p")
                    .long("http_port")
                    .value_name("HTTP_PORT")
                    .help("TCP port to listen on for HTTP")
                    .takes_value(true)
                    .required(false),
            )
            .arg(
                clap::Arg::with_name("dump_dir")
                    .short("d")
                    .long("dump_dir")
                    .value_name("DUMP_DIR")
                    .help("Directory to use to dump debug WAV streams")
                    .takes_value(true)
                    .required(false),
            )
            .arg(
                clap::Arg::with_name("model")
                    .short("m")
                    .long("model")
                    .value_name("MODEL")
                    .help("TensorFlow model to use")
                    .takes_value(true)
                    .required(true),
            )
            .arg(
                clap::Arg::with_name("alphabet")
                    .short("a")
                    .long("alphabet")
                    .value_name("ALPHABET")
                    .help("Alphabet file matching the TensorFlow model to use")
                    .takes_value(true)
                    .required(true),
            )
            .arg(
                clap::Arg::with_name("lm")
                    .short("lm")
                    .long("lm")
                    .value_name("LM")
                    .help("KenLM Language Model to use")
                    .takes_value(true)
                    .required(true),
            )
            .arg(
                clap::Arg::with_name("trie")
                    .short("trie")
                    .long("trie")
                    .value_name("TRIE")
                    .help("KenLM Trie to use")
                    .takes_value(true)
                    .required(true),
            )
            .arg(
                clap::Arg::with_name("v")
                    .short("v")
                    .multiple(true)
                    .help("Sets the level of verbosity"),
            )
            .get_matches();

        RuntimeConfig {
            http_ip: ArgsParser::to_ip_addr(matches.value_of("http_ip")),
            http_port: ArgsParser::to_port(matches.value_of("http_port")),
            dump_dir: String::from(matches.value_of("dump_dir").unwrap_or("/tmp")),
            model: String::from(matches.value_of("model").unwrap()),
            alphabet: String::from(matches.value_of("alphabet").unwrap()),
            lm: String::from(matches.value_of("lm").unwrap()),
            trie: String::from(matches.value_of("trie").unwrap()),
            verbosity_level: ArgsParser::to_verbosity_level(matches.occurrences_of("v")),
        }
    }
}

#[test]
fn test_to_ip_addr() {
    assert_eq!(
        ArgsParser::to_ip_addr(Some("")),
        IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))
    );
    assert_eq!(
        ArgsParser::to_ip_addr(Some("239.255.0.1")),
        IpAddr::V4(Ipv4Addr::new(239, 255, 0, 1))
    );
    assert_eq!(
        ArgsParser::to_ip_addr(Some("1.2.3.4")),
        IpAddr::V4(Ipv4Addr::new(1, 2, 3, 4))
    );
    assert_eq!(
        ArgsParser::to_ip_addr(Some("::1")),
        IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1))
    );
    assert_eq!(
        ArgsParser::to_ip_addr(Some("ffx3::1")),
        IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))
    );
    assert_eq!(
        ArgsParser::to_ip_addr(Some("ff03::1")),
        IpAddr::V6(Ipv6Addr::new(0xff03, 0, 0, 0, 0, 0, 0, 1))
    );
}

#[test]
fn test_to_port() {
    assert_eq!(ArgsParser::to_port(Some("xxx")), 8080);
    assert_eq!(ArgsParser::to_port(Some("8080")), 8080);
    assert_eq!(ArgsParser::to_port(Some("1234")), 1234);
}

#[test]
fn test_to_verbosity_level() {
    assert_eq!(ArgsParser::to_verbosity_level(0), VerbosityLevel::ERROR);
    assert_eq!(ArgsParser::to_verbosity_level(1), VerbosityLevel::WARN);
    assert_eq!(ArgsParser::to_verbosity_level(2), VerbosityLevel::INFO);
    assert_eq!(ArgsParser::to_verbosity_level(3), VerbosityLevel::DEBUG);
    assert_eq!(ArgsParser::to_verbosity_level(4), VerbosityLevel::DEBUG);
    assert_eq!(ArgsParser::to_verbosity_level(42), VerbosityLevel::DEBUG);
}

#[test]
fn test_args() {
    let rc = ArgsParser::from_cli();

    assert_eq!(rc.http_ip.to_string(), "0.0.0.0");
    assert_eq!(rc.http_port.to_string(), "8080");
    assert_eq!(rc.verbosity_level, VerbosityLevel::ERROR);
}
