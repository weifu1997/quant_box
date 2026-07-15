# Web workspace deployment

## Supported environments

The Web workspace uses the same Python/FastAPI backend and React production build on Windows and Ubuntu. Background jobs use `sys.executable` and repository-relative `Path` values. Process groups are created and stopped with Windows APIs on Windows and POSIX sessions/signals on Ubuntu.

## Windows local use

For development and daily desktop use, run:

```text
15_启动Web仪表盘.bat
```

This starts the FastAPI backend and Vite development server on `127.0.0.1`.

For a production-like single-server check:

```powershell
python scripts\dev_env.py sync --build-web
.\.venv\Scripts\python.exe scripts\dev_env.py doctor --strict --runtime-only
.\.venv\Scripts\python.exe scripts\run_dashboard.py --host 127.0.0.1 --port 8000
```

FastAPI serves `web/dist` at `http://127.0.0.1:8000`.

The manual-order stock detail view uses the configured Tushare proxy for the fixed `rt_k` quote call. If that service is unreachable or rejects the request, the API falls back to the latest local `data/raw/<TS_CODE>.csv` daily close and labels it as non-live; deployments should therefore persist both the private Tushare configuration and local market-data cache.

## Ubuntu local or server use

Synchronize and build during deployment:

```bash
python3.11 scripts/dev_env.py sync --build-web
.venv/bin/python scripts/dev_env.py doctor --strict --runtime-only
bash scripts/start_dashboard.sh
```

After every `git pull`, rerun the sync command before restarting systemd. It compares the Python lock, npm lock, and frontend source fingerprint, so unchanged dependencies are skipped while a changed frontend is rebuilt. `scripts/start_dashboard.sh` is intentionally validation-only: it never installs packages or builds during service startup.

The script defaults to `127.0.0.1:8000`. Override only when the network boundary is already protected:

```bash
DASHBOARD_HOST=0.0.0.0 DASHBOARD_PORT=8000 bash scripts/start_dashboard.sh
```

## Cloud security boundary

The application can run data downloads, backtests, signal workflows, and validated holdings updates. Do not expose port 8000 directly to the public Internet.

Use one of these access boundaries:

1. Keep the service on `127.0.0.1` and access it through an SSH tunnel.
2. Put it behind a private VPN such as Tailscale or WireGuard.
3. Put Nginx/Caddy in front with TLS and authentication, while FastAPI continues listening on `127.0.0.1`.

Example SSH tunnel from the local machine:

```bash
ssh -L 8000:127.0.0.1:8000 user@server
```

Then open `http://127.0.0.1:8000` locally.

## systemd example

Create `/etc/systemd/system/quant-box-dashboard.service`:

```ini
[Unit]
Description=quant_box Web workspace
After=network.target

[Service]
Type=simple
User=quantbox
WorkingDirectory=/opt/quant_box
Environment=DASHBOARD_HOST=127.0.0.1
Environment=DASHBOARD_PORT=8000
ExecStart=/bin/bash /opt/quant_box/scripts/start_dashboard.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now quant-box-dashboard
sudo systemctl status quant-box-dashboard
```

Generated `data/`, `outputs/`, private settings, account state, and holdings must be persisted and backed up separately from the Git checkout.

## Browser verification

Windows development machines use the installed Chrome channel by default:

```powershell
cd web
npm run test:e2e
```

On Ubuntu CI or a server test environment:

```bash
cd web
npx playwright install --with-deps chromium
PLAYWRIGHT_CHANNEL=chromium npm run test:e2e
```

The browser suite intercepts mutating APIs and uses temporary mock state. It verifies button wiring and request contracts without downloading market data or modifying real holdings.
