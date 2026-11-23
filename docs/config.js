// Dashboard connection settings (see demo/README.md, step 7).
//
// This file is public once the repository is on GitHub, so use a broker
// credential with the permission "Subscribe only" here, never the edge device's
// publishing credential.
window.BMS_CONFIG = {
  host: "wfa61d72.ala.us-east-1.emqxsl.com",
  port: 8084,
  username: "bms-dashboard",
  password: "bmsview-oeYHa5a9xwtSu4",
  topic: "bms-soc-estimation/demo",  // must match [demo] topic in demo/config.toml
};
