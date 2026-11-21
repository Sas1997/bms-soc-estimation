# Edge and cloud demo: Raspberry Pi 5 + HiveMQ Cloud + live dashboard

The Raspberry Pi replays a test drive cycle one second at a time, as if the measurements came from a live
Li-ion cell, and runs four SoC estimators on the device: the calibrated EKF, the LSTM and the Transformer
(as ONNX models in ONNX Runtime, no PyTorch needed), and coulomb counting. It streams measurements and
estimates over MQTT with TLS to a free HiveMQ Cloud broker. A web page (`docs/index.html`) subscribes over a
secure WebSocket and plots everything live. This mirrors a cloud-connected BMS: estimation at the edge,
logging and monitoring in the cloud.

```
 Raspberry Pi 5                         HiveMQ Cloud (free)              Browser
 demo/edge.py                           MQTT broker                      docs/index.html
 replay -> EKF, LSTM, Transformer, CC   ---- MQTT over TLS, 8883 ---->   ---- secure WebSocket, 8884 ---->
 local log in demo/runs/                retained: meta, history, summary  live charts and metrics
```

Total time: about 1 hour, most of it waiting for the SD card and the installs.

## What you need

- The Raspberry Pi 5.
- A microSD card of 32 GB (16 GB is the minimum), and a card reader for your laptop.
- A USB-C power supply. The official Raspberry Pi 27 W supply is the safe choice for a Pi 5.
- A Wi-Fi network with a plain password (home Wi-Fi or a phone hotspot), or an Ethernet cable to your router.
  The campus eduroam network needs an enterprise login that the setup below does not configure.
- Your Windows laptop on the same network.

## Step 1. Prepare the SD card (10 min)

1. Install Raspberry Pi Imager from <https://www.raspberrypi.com/software/> and open it.
2. Device: **Raspberry Pi 5**. Operating system: **Raspberry Pi OS (64-bit)**. Storage: your microSD card.
3. In the **Customisation** steps:
   - Hostname: `bms-pi`
   - Localisation: time zone `America/Toronto`, keyboard `us`
   - User: username `ali`, and a password you will remember
   - Wi-Fi: your network name and password, country `CA`
   - Remote access: turn **SSH** on, with **password authentication**
   - Raspberry Pi Connect: skip
4. Write the card and wait until Imager says it is done.

## Step 2. Start the Pi and log in (5 min)

1. Put the card in the Pi and connect power. The first start takes 2 to 3 minutes.
2. On the laptop, open **PowerShell** and run:

   ```
   ssh ali@bms-pi.local
   ```

   Type `yes` when asked about the fingerprint, then your Pi password. The prompt changes to `ali@bms-pi:~ $`.
   This window is now a terminal on the Pi.

## Step 3. Copy the project to the Pi (1 min)

Open a **second** PowerShell window on the laptop (not the one logged in to the Pi) and run:

```
scp "$env:USERPROFILE\Downloads\bms-soc-estimation-repo.zip" ali@bms-pi.local:~/
```

If the repository is public on GitHub, you can instead run `git clone https://github.com/SAS1997/bms-soc-estimation.git` on the Pi and skip the `unzip` line in step 4.

## Step 4. Install and test (10 min)

In the Pi window:

```
sudo apt update
sudo apt install -y python3-venv unzip
unzip -q bms-soc-estimation-repo.zip
cd bms-soc-estimation
python3 -m venv ~/bms-venv
source ~/bms-venv/bin/activate
pip install -r demo/requirements-pi.txt
python -m pytest -q
```

The last line should end with `47 passed, 3 skipped` (the three skipped tests need PyTorch or h5py, which the
Pi does not need). After you log in again later, run `cd bms-soc-estimation` and
`source ~/bms-venv/bin/activate` before any `python` command.

## Step 5. Measure the estimators on the Pi (2 min)

```
python experiments/m4_cost.py --label raspberry-pi-5
python demo/edge.py --offline --cycle 10degC_US06 --speed 1000
```

The first command times every estimator sample by sample and prints one line per estimator. The second runs
the whole edge loop on the 10 °C US06 cycle without network (under a minute) and prints a summary table. Keep both outputs.

## Step 6. Create the cloud broker (10 min)

1. Go to <https://console.hivemq.cloud>, sign up, and choose the **Serverless** plan (free, no credit card).
   Create a cluster.
2. Open the cluster details and copy the **Cluster URL** (it ends in `.hivemq.cloud`). The ports are **8883**
   for MQTT and **8884** for WebSocket.
3. Open **Access Management**, then **Credentials**, and add two:
   - username `bms-pi`, a password, permission **Publish and Subscribe** (for the Pi)
   - username `bms-dashboard`, a different password, permission **Subscribe Only** (for the web page)

On the free plan a permission applies to every topic on the cluster, so use this cluster for this demo only.

## Step 7. Point the Pi at the broker (2 min)

In the Pi window:

```
cp demo/config.example.toml demo/config.toml
nano demo/config.toml
```

Replace the three example values: `host` with the Cluster URL (no `https://`), `username` with `bms-pi`, and
`password` with its password. Save with `Ctrl+O`, `Enter`, then exit with `Ctrl+X`. This file stays on the Pi:
git ignores it.

## Step 8. Point the dashboard at the broker (2 min)

On the laptop:

1. Unzip `bms-soc-estimation-repo.zip` (right-click, Extract All).
2. Open `docs\config.js` in Notepad. Fill in `host` (the Cluster URL), `username: "bms-dashboard"` and its
   password. Save.
3. Double-click `docs\index.html`. It opens in your browser and should show **Cloud broker: connected**.

## Step 9. Run the demo

In the Pi window:

```
python demo/edge.py
```

This replays the 25 °C UDDS test cycle at 10 times real time (about 27 minutes), with the EKF and coulomb
counting started 20 points below the true SoC. The dashboard fills in live; the Pi prints a status line every
10 seconds and a summary at the end.

Other runs:

| Command | What it shows | Duration |
|:--|:--|--:|
| `python demo/edge.py --cycle n20degC_UDDS` | -20 °C, the hardest temperature | about 14 min |
| `python demo/edge.py --cycle 10degC_US06` | 10 °C, the temperature no model saw | about 7 min |
| `python demo/edge.py --speed 1` | real time, one sample per second | about 4.4 h |
| `python demo/edge.py --soc-error 0` | correct initial SoC | |

For a long run that should continue after you close the window:

```
nohup python demo/edge.py --speed 1 > edge.log 2>&1 &
```

`Ctrl+C` stops a run cleanly. Each run is logged in `demo/runs/` on the Pi.

## Step 10. Put the dashboard online (optional, 5 min)

GitHub Pages serves `docs/` as a public web page for free when the repository is public.

1. Commit `docs/config.js` with the **subscribe-only** credential (never the `bms-pi` one) and push.
2. On GitHub: **Settings**, **Pages**, Source **Deploy from a branch**, Branch **main**, folder **/docs**, Save.
3. After about a minute the dashboard is at <https://sas1997.github.io/bms-soc-estimation/>.

The broker keeps the last run (retained messages), so the page shows the most recent run with its metrics even
when the Pi is off.

## Troubleshooting

| Problem | Fix |
|:--|:--|
| `ssh: Could not resolve hostname bms-pi.local` | Wait 2 more minutes after power-on. Check the laptop is on the same network. Otherwise find the Pi's IP address in your router's device list and use `ssh ali@` followed by that address. |
| The Pi never joins Wi-Fi | Write the card again with the exact network name and password (both are case sensitive). |
| `MQTT connection failed: Not authorized` | Username or password in `demo/config.toml` does not match the `bms-pi` credential. |
| `MQTT connection failed` with a certificate or TLS error | The Pi's clock is wrong. Run `timedatectl`; it should say `System clock synchronized: yes` once the Pi is online. |
| Dashboard says `not configured` | `docs/config.js` still has an empty `host` or `username`. |
| Dashboard stays on `reconnecting` | Wrong Cluster URL or password in `docs/config.js`, or a network that blocks port 8884. |
| `pip install` is slow | Normal on the first install. |
