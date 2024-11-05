import socket

def get_LAN_ip():
    try:
        # Create a temporary socket connection to determine the local IP address
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # IP here is arbitrary since we're not actually sending data
        temp_socket.connect(("8.8.8.8", 80))
        lan_ip = temp_socket.getsockname()[0]
        temp_socket.close()
        return lan_ip
    except Exception as e:
        print("Could not determine LAN IP:", e)
        return None

# Example usage
def get_cam_link():
    return f'{get_LAN_ip()}:8000/video_feed'

def get_docs_link():
    return f'{get_LAN_ip()}:8000/docs'

