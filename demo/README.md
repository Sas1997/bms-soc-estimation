# Edge and cloud demo: Raspberry Pi 5 + HiveMQ Cloud + live dashboard

The Raspberry Pi replays a test drive cycle one second at a time, as if the measurements came from a live
Li-ion cell, and runs four SoC estimators on the device: the calibrated EKF, the LSTM and the Transformer
(as ONNX models in ONNX Runtime, no PyTorch needed), and coulomb counting. It streams measurements and
estimates over MQTT with TLS to a free HiveMQ Cloud broker. A web page (`docs/index.html`) subscribes over a
secure WebSocket and plots everything live. This mirrors a cloud-connected BMS: estimation at the edge,
logging and monitoring in the cloud. Instead of a replay, the Pi can also read a real smart battery pack over
its SMBus data port (step 11).

```
 Raspberry Pi 5                         HiveMQ Cloud (free)              Browser
 demo/edge.py                           MQTT broker                      docs/index.html
 replay -> EKF, LSTM, Transformer, CC   ---- MQTT over TLS, 8883 ---->   ---- secure WebSocket, 8884 ---->
 local log in demo/runs/                retained: meta, history, summary  live charts and metrics
```

Total time: about 1 hour, most of it waiting for the SD card and the installs.

## What you need

- The Raspberry Pi 5.
- A microSD card of 32 GB (16 GB is the minimum).
- A USB-C power supply. The official Raspberry Pi 27 W supply is the safe choice for a Pi 5.
- A Wi-Fi network with a plain password (home Wi-Fi or a phone hotspot), or an Ethernet cable to your router.
  The campus eduroam network needs an enterprise login that the setup below does not configure.
- Your Windows laptop on the same network.
- To write the card: a card reader for the laptop (step 1A), or, without one, a USB keyboard and mouse, a
  monitor with a micro-HDMI-to-HDMI cable, and an Ethernet cable to a router or to the laptop (step 1B).

## Step 1A. Prepare the SD card on the laptop (10 min)

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

## Step 1B. No card reader: install over the network from the Pi itself (30 min)

The Pi 5 can download Raspberry Pi Imager into its memory and write the card in its own slot. This needs a
wired internet connection, a keyboard and a monitor on the Pi.

1. **Give the Pi a wired connection.** Plug an Ethernet cable from the Pi into your router. Or, if the laptop
   has an Ethernet port (or a USB Ethernet adapter), share the laptop's Wi-Fi over the cable: press
   `Win + R`, type `ncpa.cpl`, Enter; right-click **Wi-Fi**, **Properties**, **Sharing** tab; tick **Allow other
   network users to connect through this computer's Internet connection**; under **Home networking connection**
   choose **Ethernet**; OK. Then plug the cable from the laptop into the Pi.
2. Put the blank microSD card in the Pi. Remove any USB drives. Connect the keyboard, mouse and monitor (use the
   micro-HDMI port next to the USB-C power port).
3. **Hold Shift** on the Pi's keyboard while you connect the power, and keep holding until the screen says it is
   downloading the installer. If the Pi is already showing its bootloader screen, holding Shift there also
   works. If it asks you to confirm, press Space.
4. After a few minutes Raspberry Pi Imager opens on the Pi's screen. Choose and customise exactly as in step 1A
   (device, OS, storage = the SD card in the Pi, hostname `bms-pi`, user `ali`, Wi-Fi, SSH on). If the Pi will
   stay on the laptop's cable, you can skip Wi-Fi.
5. Write, confirm, and wait (about 10 to 20 minutes; the OS is downloaded first). When Imager says it is done,
   let the Pi restart. It boots from the new card; the first start takes 2 to 3 minutes.

If the installer waits for the network: check the cable, and on the laptop untick and re-tick the sharing box.

## Step 2. Start the Pi and log in (5 min)

1. Step 1A: put the card in the Pi and connect power. The first start takes 2 to 3 minutes. (Step 1B: the Pi
   is already running.)
2. On the laptop, open **PowerShell** and run:

   ```
   ssh ali@bms-pi.local
   ```

   Type `yes` when asked about the fingerprint, then your Pi password. The prompt changes to `ali@bms-pi:~ $`.
   This window is now a terminal on the Pi.

   If `bms-pi.local` is not found, open a terminal on the Pi itself (if it has a screen), run `hostname -I`,
   and use that address instead: `ssh ali@192.168.137.23`, for example (addresses starting with 192.168.137
   come from the laptop's shared connection).

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

The last line should end with `66 passed, 3 skipped` (the three skipped tests need PyTorch or h5py, which the
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

## Step 11 (optional). Read a real smart battery over SMBus

No pack yet? Everything below also runs on an emulated pack:
`python demo/smbus_log.py --emulate 25degC_UDDS --identify` and
`python demo/edge.py --source smbus-emulated` (the dashboard then shows the gauge as a sixth tile and line).

**Safety first.** The software only reads (no write commands exist in it), but the wiring is yours:

- Connect only three wires: the pack's SMBus data (SDA, sometimes labelled SMD), clock (SCL, SMC) and ground.
  Never connect the pack's power terminals to the Pi. Find the pins in the pack's or the device's documentation,
  not by probing.
- Connect ground first.
- The Pi's I2C pins are 3.3 V only and already have 1.8 kΩ pull-ups to 3.3 V. If the pack pulls its SMBus lines
  up to a higher voltage (measure SDA and SCL to ground with the Pi disconnected), put a bidirectional I2C level
  shifter in between.
- Some packs keep their gauge asleep or their output off until they see a charger or a host; that is normal.

**Wiring** (Raspberry Pi 5 header): pin 3 = GPIO 2 = SDA, pin 5 = GPIO 3 = SCL, pin 6 = ground.

**Enable I2C and look for the battery** (smart batteries answer at address 0x0b):

```
sudo raspi-config nonint do_i2c 0
sudo apt install -y i2c-tools
i2cdetect -y -r 1 0x0b 0x0b
```

`-r` makes `i2cdetect` probe with a read, and the address range limits it to the battery, so nothing else on the
bus is touched. `0b` in the grid means the battery answered.

**Log the pack and identify its vendor registers** (1 Hz for 30 minutes; run the pack through charge and
discharge if you can, which separates quantities that otherwise move together):

```
python demo/smbus_log.py --bus 1 --duration 1800 --scan 0x24-0x3f --identify
```

It prints the pack's name, chemistry and design capacity, then every 10 samples voltage, current, temperature and
the gauge's SoC. At the end, `demo/runs/smbus-<time>/registers.md` lists what each scanned code tracks.

**Stream it to the dashboard**, for a pack with, for example, 3 cells in series and 2 in parallel:

```
python demo/edge.py --source smbus --series 3 --parallel 2
```

The pack's own fuel gauge is then the reference. Coulomb counting uses the pack's full-charge capacity; the EKF
and the networks are models of an LG HG2 cell, so their numbers are meaningful only for a pack built from that
cell. If the logger found a cell-voltage register, pass it with `--cell-voltage-cmd 0x3f` (for example).

If reads fail intermittently, slow the bus: add `dtparam=i2c_arm_baudrate=50000` to `/boot/firmware/config.txt`
and reboot.

## Step 12 (optional). Read a TI BQ27426 fuel gauge

The BQ27426 is a single-cell system-side Impedance Track gauge (TI's evaluation board: BQ27426EVM-738). The cell
and the load connect to the board, so the gauge measures the cell's current through its sense resistor; the Pi
only reads the gauge over I2C. No board yet? `python demo/edge.py --source bq27426-emulated` runs the same code on
an emulated gauge.

1. **Configure the gauge for the cell first**, with TI's Battery Management Studio and an EV2400 adapter as in
   the EVM user's guide: design capacity, terminate voltage, and the chemistry profile (for a 4.2 V cell such as
   the LG HG2: CHEM_B, Chem ID 1202). This repository only reads the gauge and never changes its settings.
2. **Wire the board's I2C header to the Pi**: SDA to pin 3, SCL to pin 5, ground to pin 6, ground first. As in
   step 11, if the board pulls SDA and SCL above 3.3 V, use a level shifter.
3. **Check that it answers** at address 0x55: `i2cdetect -y -r 1 0x55 0x55`.
4. **Log it** (one burst read per second; the gauge allows at most two commands per second):

   ```
   python demo/smbus_log.py --gauge bq27426 --bus 1 --duration 3600
   ```

   The log has the gauge's StateOfCharge (filtered and unfiltered), capacities, StateOfHealth with its Qmax
   learning status, and flags. At the end it prints a sign check; `current_sign_agrees_with_DSG` should be 1.0.

5. **Stream it**, with the gauge's SoC as the reference: `python demo/edge.py --source bq27426`.

The temperature the gauge reports may be its internal sensor rather than the cell's (OpConfig [TEMPS] in the
manual), which matters for the EKF and the networks at low temperature.

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
| The Pi switches itself off (red light) after a few seconds or minutes | The power supply cannot hold 5 V under load. Use the official Raspberry Pi 27 W USB-C supply (or another good USB-C supply with a USB-C cable, not a USB-A adapter). Check with `vcgencmd pmic_read_adc EXT5V_V`: about 5.1 V is healthy. |
| `smbus_log.py`: `Remote I/O error` | No answer at 0x0b: check SDA/SCL are not swapped, ground is connected, and the pack is awake (step 11). |
| `smbus_log.py`: `PEC mismatch` keeps repeating | Noise on the bus: shorter wires, slower bus (step 11), or `--pec off` to see whether the gauge sends PEC at all. |
