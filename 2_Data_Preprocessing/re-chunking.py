import os
import json
import re
import requests
import argparse
from tqdm import tqdm

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

def clean_markdown(text):
    """清理文本中的 Markdown 符號"""
    # 移除代碼塊
    text = re.sub(r'```[\s\S]*?```', '', text)
    # 移除 HTML 標籤
    text = re.sub(r'<[^>]+>', '', text)
    # 移除粗體、斜體等格式
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'__(.*?)__', r'\1', text)  # __bold__
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # *italic*
    text = re.sub(r'_(.*?)_', r'\1', text)  # _italic_
    text = re.sub(r'~~(.*?)~~', r'\1', text)  # ~~strikethrough~~
    # 移除連結
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)  # [text](url)
    # 移除圖片
    text = re.sub(r'!\[(.*?)\]\((.*?)\)', r'\1', text)  # ![alt](url)
    # 移除標題符號
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  # # heading
    # 移除列表符號
    text = re.sub(r'^[\-\*\+]\s+', '', text, flags=re.MULTILINE)  # - list
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)  # 1. list
    # 移除水平線
    text = re.sub(r'^\-{3,}$|^\*{3,}$|^_{3,}$', '', text, flags=re.MULTILINE)
    return text.strip()

def step1_sub_chunking(text):
    """DP6 步驟 1: 將原始文本分割成多個子塊 (Sub-chunks)"""
    print(f"[DEBUG] Step 1: 開始進行 Sub-chunking，輸入文本長度: {len(text)} 字符")
    prompt = f"""Segment the text provided below into multiple sub-chunks. Each sub-chunk should:
- Focus on a single, specific topic.
- Contain fewer than 4096 characters.
- Be separated by a unique delimiter: ---CHUNK_BOUNDARY---

Here is the text to segment:
{text}"""
    
    response = query_ollama(prompt)
    if not response:
        print(f"[ERROR] Step 1: API 回應為空")
        return []
    # 根據分隔符切分並過濾掉太短的無效片段
    chunks = [c.strip() for c in response.split("---CHUNK_BOUNDARY---") if len(c.strip()) > 30]
    print(f"[DEBUG] Step 1: 分割後得到 {len(chunks)} 個有效的 sub-chunks")
    return chunks

def step2_generate_title(sub_chunk):
    """DP6 步驟 2: 為每個子塊生成簡潔的標題"""
    print(f"[DEBUG] Step 2: 正在為 sub-chunk 生成標題 (長度: {len(sub_chunk)} 字符)")
    prompt = f"""Based on the following paragraph, generate a concise and meaningful English title. 
Respond ONLY with the title text, nothing else.

Paragraph: 
{sub_chunk}"""
    
    title = query_ollama(prompt)
    if title:
        # 清洗標題，移除可能的引號或多餘文字
        title = re.sub(r'["\']', '', title).replace('Title:', '').strip()
        print(f"[DEBUG] Step 2: 生成的標題: {title}")
    else:
        print(f"[ERROR] Step 2: 無法生成標題")
    return title or "Untitled Chunk"

def clean_markdown(text):
    """清理文本中的 Markdown 符號"""
    # 移除代碼塊
    text = re.sub(r'```[\s\S]*?```', '', text)
    # 移除 HTML 標籤
    text = re.sub(r'<[^>]+>', '', text)
    # 移除粗體、斜體等格式
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'__(.*?)__', r'\1', text)  # __bold__
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # *italic*
    text = re.sub(r'_(.*?)_', r'\1', text)  # _italic_
    text = re.sub(r'~~(.*?)~~', r'\1', text)  # ~~strikethrough~~
    # 移除連結
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)  # [text](url)
    # 移除圖片
    text = re.sub(r'!\[(.*?)\]\((.*?)\)', r'\1', text)  # ![alt](url)
    # 移除標題符號
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  # # heading
    # 移除列表符號
    text = re.sub(r'^[\-\*\+]\s+', '', text, flags=re.MULTILINE)  # - list
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)  # 1. list
    # 移除水平線
    text = re.sub(r'^\-{3,}$|^\*{3,}$|^_{3,}$', '', text, flags=re.MULTILINE)
    # 移除 "Chunk N:" 前綴
    text = re.sub(r'^Chunk\s+\d+:\s*', '', text, flags=re.MULTILINE)
    return text.strip()

def process_rechunking(input_dir, output_file):
    """整合執行流程：Sub-chunk -> Regenerate Title -> Extract (處理 JSON 格式的輸入)"""
    all_chunks_data = []
    
    # 獲取所有 JSON 檔案
    print(f"[DEBUG] 開始掃描目錄: {input_dir}")
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"[INFO] 找到 {len(files)} 個 JSON 檔案")
    
    if len(files) == 0:
        print(f"[ERROR] 在 {input_dir} 中找不到任何 .json 檔案")
        return
    
    print(f"開始執行 DP6 流程...\n")

    for file_name in tqdm(files, desc="Processing Files"):
        file_path = os.path.join(input_dir, file_name)
        print(f"[DEBUG] 正在處理文件: {file_name}")
        
        try:
            # 讀取 JSON 檔案
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            print(f"[DEBUG] 成功讀取 JSON，包含 {len(json_data)} 個條目")
        except Exception as e:
            print(f"[ERROR] 無法讀取 JSON 檔案: {e}")
            continue
        
        # 處理 JSON 中的每一個條目
        for idx, entry in enumerate(json_data):
            print(f"[DEBUG] 處理第 {idx + 1}/{len(json_data)} 個條目")
            paper_title = entry.get('title', 'Unknown')
            section_title = entry.get('section_title', 'Section')
            content = entry.get('content', '')
            
            if not content:
                print(f"[WARN] 條目 {idx + 1} 沒有 content，跳過")
                continue
            
            print(f"[DEBUG] 標題: {paper_title[:50]}...")

            # Step 1: Sub-chunking
            sub_chunks = step1_sub_chunking(content)
            n = len(sub_chunks)
            
            for i, sc in enumerate(sub_chunks, 1):
                print(f"[DEBUG] 處理 Chunk {i}/{n}")
                # Step 2: Regenerate Title
                topic = step2_generate_title(sc)
                
                # 清理 content 中的 markdown 符號
                cleaned_content = clean_markdown(sc)
                
                # 調整 section_title 為 "original_section\nsection_title" 格式
                combined_section_title = f"{section_title}\n{topic}"
                
                # 存入最終 JSON 格式（不包含 formatted_output）
                all_chunks_data.append({
                    "id": len(all_chunks_data) + 1,
                    "paper": paper_title,
                    "section_title": combined_section_title,
                    "content": cleaned_content
                })
                print(f"[DEBUG] 已存入 chunk (ID: {len(all_chunks_data)})")

    # 儲存結果
    print(f"[DEBUG] 開始儲存結果至: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(all_chunks_data, out_f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] 結果已成功儲存")
    except Exception as e:
        print(f"[ERROR] 無法儲存結果: {e}")
        return
    
    print(f"\n[INFO] 處理完成！總共生成 {len(all_chunks_data)} 個 chunks，已儲存至 {output_file}")

if __name__ == "__main__":
    print("="*60)
    print("[START] DP6: Advanced-identifier Re-chunking Tool")
    print("="*60)
    
    parser = argparse.ArgumentParser(description="DP6: Advanced-identifier Re-chunking Tool")
    parser.add_argument("--input_dir", required=True, help="存放原始 JSON 檔案的目錄")
    parser.add_argument("--output_file", required=True, help="輸出的 JSON 檔案路徑")
    
    args = parser.parse_args()
    print(f"[INFO] 輸入目錄: {args.input_dir}")
    print(f"[INFO] 輸出文件: {args.output_file}")
    print("="*60 + "\n")
    
    try:
        process_rechunking(args.input_dir, args.output_file)
    except Exception as e:
        print(f"[FATAL ERROR] 程序崩潰: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("[END] 程序執行完成")
    print("="*60)