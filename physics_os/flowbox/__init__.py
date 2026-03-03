"""FlowBox — Navier-Stokes 2D periodic-box product.

Commercial product slice built on the Ontic Engine Runtime.
Provides preset-based deterministic NS2D simulations with
MP4 render, sanitized fields, metrics, and Ed25519 certificates.

Usage (API)::

    POST /v1/flowbox/run
    {
      "preset": "taylor_green",
      "grid": 512,
      "steps": 500,
      "seed": 42
    }

Usage (SDK)::

    from physics_os.sdk.client import OnticClient

    client = OnticClient(api_key="sk-...")
    job = client.flowbox("taylor_green", grid=512, steps=500)
    open("sim.mp4", "wb").write(client.flowbox_render(job.job_id))

Usage (CLI)::

    physics_os flowbox --preset taylor_green --grid 512 --steps 500 -o result.json
"""

__version__ = "1.0.0"
PRODUCT_KEY = "flowbox"
