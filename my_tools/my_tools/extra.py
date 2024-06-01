import re


def get_local_ace_url(
        stream_id: str, docker_port: int = 8000, server_ip: str = "127.0.0.1"
) -> str:
    """
    To watch ace streams:

        1. Start docker daemon in terminal or just open desktop docker app.
        2. Pull docker image:
            docker pull ikatson/aceproxy:latest
        3. Run docker image:
            docker run -t -p 8000:8000 ikatson/aceproxy
        4. Get ace stream id, possibly from stream url in format acestream://{stream_id}.
        5. Build local stream url using this function.
        6. Go to VLC player -> Open Network -> enter url -> watch.

    Example url:
    http://127.0.0.1:8000/pid/b28db77c5084da7993395d77df96c30bb134f0a9/stream.mp4
    """
    url = f"http://{server_ip}:{docker_port}/pid/{stream_id}/stream.mp4"
    return url
