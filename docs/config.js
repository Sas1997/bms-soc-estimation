// Dashboard connection settings (see demo/README.md, step 7).
//
// This file is public once the repository is on GitHub, so use a broker
// credential with the permission "Subscribe only" here, never the edge device's
// publishing credential.
window.BMS_CONFIG = {
  host: "",                          // broker address, without https://
  port: 8084,                        // MQTT over secure WebSocket (EMQX Serverless: 8084)
  path: "/mqtt",
  username: "",                      // subscribe-only username
  password: "",                      // subscribe-only password
  topic: "bms-soc-estimation/demo",  // must match [demo] topic in demo/config.toml
};
