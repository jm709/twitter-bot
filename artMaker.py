import random
import requests
import time

def generate_image(prompt):
    api_key = ""
    headers = {
        "apikey": api_key
    }
    payload = {
        "prompt": prompt,
        "params": {
            "n": 1,
            "width": 512,
            "height": 512,
            "steps": 20,
            "sampler_name": "k_euler"
        },
        "nsfw": False,
        "censor_nsfw": True,
        "models": ["stable_diffusion"],
    }
    response = requests.post("https://stablehorde.net/api/v2/generate/async", headers=headers, json=payload)
    response_data = response.json()
    print(response_data)
    request_id = response_data['id']
    print(f"Submitted request ID: {request_id}")

    status_url = f"https://stablehorde.net/api/v2/generate/check/{request_id}"

    while True:
        status = requests.get(status_url, headers=headers).json()
        if status['done']:
            break
        print(f"Waiting... {status['wait_time']}s remaining")
        time.sleep(5)

    result_url = f"https://stablehorde.net/api/v2/generate/status/{request_id}"
    result_data = requests.get(result_url, headers=headers).json()

    print(result_data)
    for idx, gen in enumerate(result_data['generations']):
        # If you wanted to improve this you would make sure you don't overwrite another image...
        randint = random.randint(10000000, 99999999)
        image_url = gen['img']
        image = requests.get(image_url)
        with open(f"image_{randint}.png", "wb") as f:
            f.write(image.content)
        print(f"Saved image_{randint}.png")

    return f"image_{randint}"


