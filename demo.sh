ffmpeg -re -stream_loop -1 -i videos/cctv.mp4 -c:v copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:3554/live
