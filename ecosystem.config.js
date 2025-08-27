module.exports = {
  apps: [{
    name: "fastapi",
    interpreter: "/root/Fast_API_dialog/.venv/bin/python",
    script: "/root/Fast_API_dialog/.venv/bin/gunicorn",
    args: "-w 1 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000 --timeout 120",
    env: {
      CUDA_VISIBLE_DEVICES: "0",
      TF_FORCE_GPU_ALLOW_GROWTH: "true",
      PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
    },
    exec_mode: "fork",
    watch: false
  }]
}