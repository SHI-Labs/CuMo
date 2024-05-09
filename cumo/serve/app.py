import sys
import os
import argparse
import time
import subprocess

import cumo.serve.gradio_web_server as gws

# Execute the pip install command with additional options
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flash-attn', '--no-build-isolation', '-U'])

def start_controller():
    print("Starting the controller")
    controller_command = [
        sys.executable,
        "-m",
        "cumo.serve.controller",
        "--host",
        "0.0.0.0",
        "--port",
        "10000",
    ]
    print(controller_command)
    return subprocess.Popen(controller_command)

def start_worker(model_path: str, bits=16):
    print(f"Starting the model worker for the model {model_path}")
    model_name = model_path.strip("/").split("/")[-1]
    assert bits in [4, 8, 16], "It can be only loaded with 16-bit, 8-bit, and 4-bit."
    if bits != 16:
        model_name += f"-{bits}bit"
    worker_command = [
        sys.executable,
        "-m",
        "cumo.serve.model_worker",
        "--host",
        "0.0.0.0",
        "--controller",
        "http://localhost:10000",
        "--model-path",
        model_path,
        "--model-name",
        model_name,
        "--use-flash-attn",
    ]
    if bits != 16:
        worker_command += [f"--load-{bits}bit"]
    print(worker_command)
    return subprocess.Popen(worker_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--controller-url", type=str, default="http://localhost:10000")
    parser.add_argument("--concurrency-count", type=int, default=5)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="reload", choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    gws.args = parser.parse_args()
    gws.models = []

    gws.title_markdown += """
ONLY WORKS WITH GPU! By default, we load the model with 4-bit quantization to make it fit in smaller hardwares.
"""

    print(f"args: {gws.args}")
    model_path = gws.args.model_path
    print(model_path)
    bits = gws.args.bits
    print(bits)
    concurrency_count = int(os.getenv("concurrency_count", 5))

    controller_proc = start_controller()
    worker_proc = start_worker(model_path, bits=bits)

    # Wait for worker and controller to start
    time.sleep(10)

    exit_status = 0
    try:
        demo = gws.build_demo(embed_mode=False, concurrency_count=concurrency_count)
        demo.queue(
            status_update_rate=10,
            api_open=False
        ).launch(
            server_name=gws.args.host,
            server_port=gws.args.port,
            share=gws.args.share
        )

    except Exception as e:
        print(e)
        exit_status = 1
    finally:
        worker_proc.kill()
        controller_proc.kill()

        sys.exit(exit_status)