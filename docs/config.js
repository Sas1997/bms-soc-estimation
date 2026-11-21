// Dashboard connection settings (see demo/README.md, step 7).
//
// This file is public once the repository is on GitHub, so use a HiveMQ Cloud
// credential with the permission "Subscribe only" here, never the edge device's
// publishing credential.
window.BMS_CONFIG = {
  host: "",                          // Cluster URL from the HiveMQ console, without https://
  port: 8884,                        // MQTT over secure WebSocket
  path: "/mqtt",
  username: "",                      // subscribe-only username
  password: "",                      // subscribe-only password
  topic: "bms-soc-estimation/demo",  // must match [demo] topic in demo/config.toml
};
