Build
=====
 - Download compatible version of `native_client.tar.xz` (check `deepspeech-rs`)
 - `LB_LIBRARY_PATH=... LIBRARY_PATH=... cargo build` with both path pointing to the extracted `native_client.tar.xz`

Run
===
 - Download compatible DeepSpeech model and extract
 - 
```
$ LD_LIBRARY_PATH=...: ./target/debug/ds-srv --model models/output_graph.pbmm --lm models/lm.binary --trie models/trie -vvvvv
05:16:57 [DEBUG] ds_srv: Parsed all CLI args: RuntimeConfig { http_ip: V6(::), http_port: 8080, dump_dir: "/tmp", warmup_dir: "", warmup_cycles: 10, model: "models/output_graph.pbmm", lm: "models/lm.binary", trie: "models/trie", verbosity_level: DEBUG }
Started all thread.
05:16:57 [INFO] Inference thread started
TensorFlow: v1.11.0-rc2-4-g77b7b17
05:16:57 [INFO] Building server http://[::]:8080
DeepSpeech: v0.2.1-alpha.1-0-gae2cfe0
05:16:57 [INFO] Listening on http://[::]:8080
2018-09-27 07:16:57.157376: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
05:16:57 [DEBUG] tokio_reactor::background: starting background reactor
05:17:02 [INFO] Model ready and waiting for data to infer ...
```

Test
====

Using `4507-16021-0012.wav` from DeepSpeech's release:

```
$ curl -v -H 'Content-Type: application/octet-stream' --data-binary @"./audio/4507-16021-0012.wav" http://127.0.0.1:8080
*   Trying 127.0.0.1...
* TCP_NODELAY set
* Connected to 127.0.0.1 (127.0.0.1) port 8080 (#0)
> POST / HTTP/1.1
> Host: 127.0.0.1:8080
> User-Agent: curl/7.58.0
> Accept: */*
> Content-Type: application/octet-stream
> Content-Length: 87564
> Expect: 100-continue
>
< HTTP/1.1 100 Continue
* We are completely uploaded and fine
< HTTP/1.1 200 OK
< content-type: application/json
< content-length: 84
< date: Thu, 27 Sep 2018 05:12:36 GMT
<
* Connection #0 to host 127.0.0.1 left intact
{"status":"ok","data":[{"text":"why should one hall on the way ","confidence":1.0}]}
```
