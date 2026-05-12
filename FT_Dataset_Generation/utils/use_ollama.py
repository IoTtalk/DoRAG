import requests

def query_ollama(prompt, model='deepseek-r1:70b', api_key='6M3^TRW^7#Z4yZjj@2u3SkAY&A^EMmvpwW&AWmuFPULtEfQ3'):
    """使用您提供的最新 API 介面進行呼叫"""
    url = 'http://140.113.164.118:32159/api/generate'
    headers = {'Content-Type': 'application/json', "Authorization": f"Bearer {api_key}"}
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        "keep_alive": -1
    }

    try:
        print(f"[DEBUG] 正在向 Ollama API 發送請求...")
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        print(f"[DEBUG] API 響應狀態碼: {response.status_code}")
        data = response.json()
        full_text = data.get('response', '')
        print(f"[DEBUG] 收到回應，長度: {len(full_text)} 字符")
        # 提取 DeepSeek <think> 標籤之後的內容
        if "</think>" in full_text:
            result = full_text.split("</think>")[-1].strip()
            print(f"[DEBUG] 提取 </think> 後的內容，長度: {len(result)} 字符")
            return result
        print(f"[DEBUG] 未找到 </think> 標籤，返回全部內容")
        return full_text.strip()
    except Exception as e:
        print(f"[ERROR] Ollama API 呼叫失敗: {e}")
        import traceback
        traceback.print_exc()
        return None
