module.exports = {
  apps: [{
    name: "fastapi",
    interpreter: "/root/Fast_API_dialog/.venv/bin/python",
    script: "/root/Fast_API_dialog/.venv/bin/gunicorn",
    args: "-w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000 --timeout 120",
    env: {
      CUDA_VISIBLE_DEVICES: "0"
    },
    exec_mode: "fork",
    watch: false
  }]
}