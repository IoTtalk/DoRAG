import requests

def query_ollama(prompt, model='deepseek-r1:70b', api_key='6M3^TRW^7#Z4yZjj@2u3SkAY&A^EMmvpwW&AWmuFPULtEfQ3'):
    url = 'http://140.113.164.118:32159/api/generate'
    headers = {'Content-Type': 'application/json', "Authorization": f"Bearer {api_key}"}
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,  # set to True if you want to handle streamed responses
        "keep_alive": -1
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        return data.get('response')
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return None
