from selenium import webdriver
from selenium.webdriver.remote.webdriver import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from selenium.webdriver.common.keys import Keys
import sys
import random
import re

import os
import time

class wait_for_text_to_stabilize:
    def __init__(self, locator, timeout=10):
        self.locator = locator
        self.timeout = timeout  # Time (in seconds) to wait for text to stabilize

    def __call__(self, driver):
        element = driver.find_element(*self.locator)
        current_text = element.text
        
        # Wait for the text to stop changing
        WebDriverWait(driver, self.timeout).until(
            lambda d: element.text == current_text and len(element.text) > 0
        )
        
        # After text stabilizes, return the element
        return element


gpt_url = 'https://chatgpt.com/'

def sanitize_text(text):
    # 1. 移除控制字元（除了 \n \t）
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)

    # 2. 移除或替換隱形 Unicode 控制符號
    invisible_chars = [
        '\u200b', '\u200c', '\u200d',  # zero width
        '\u200e', '\u200f',            # direction marks
        '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',  # bidi control
        '\u2060', '\u2066', '\u2067', '\u2068', '\u2069',  # word joiners/isolate
        '\ufeff',                      # BOM
        '\u3000',                      # full-width space
    ]
    for ch in invisible_chars:
        text = text.replace(ch, ' ')

    # 3. 替換特殊空格與連字號
    replacements = {
        '\u202f': ' ',
        '\u00a0': ' ',
        '\u2010': '-',
        '\u2011': '-',
        '\u2013': '-',
        '\u2014': '-',
        '\u2028': ' ',
        '\u2029': ' ',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # 4. 替換智慧引號
    smart_quotes = {'“': '"', '”': '"', '‘': "'", '’': "'"}
    for bad, good in smart_quotes.items():
        text = text.replace(bad, good)

    # 5. 把換行統一成空格（防止 Selenium 模擬 Enter）
    text = text.replace('\n', ' ').replace('\r', ' ')

    # 6. 移除非 BMP 字元（emoji 等）
    text = ''.join(c if ord(c) <= 0xFFFF else ' ' for c in text)

    # 7. 壓縮多重空白
    text = re.sub(r' {2,}', ' ', text).strip()

    return text

class gptParser:
    def __init__(self,
                 driver,
                 gpt_url: str = gpt_url):
       
        # Start a webdriver instance and open ChatGPT
        self.driver = driver
        self.driver.get(gpt_url)

    @staticmethod
    def get_driver():
        uc.TARGET_VERSION = 133
        options = uc.ChromeOptions()        

        #driver = uc.Chrome()
        #driver = uc.Chrome(options=options)
        
        if random.choice([True, False]):
            print("Initializing Chrome driver with options...")
            options.add_argument("--incognito")
            driver = uc.Chrome(options=options)
        else:
            print("Initializing Chrome driver without options...")
            driver = uc.Chrome()

        return driver

    def __call__(self, msg: str, chunk_size = 10000):
        # Find the input field and send a question
        input_field = self.driver.find_elements(By.ID, 'prompt-textarea')[0]
        
        for i in range(0, len(msg), chunk_size):
            input_field.send_keys(sanitize_text(msg[i:i+chunk_size]))
        time.sleep(5)

        input_field.send_keys(Keys.RETURN)
        
        try:
            all_elements = self.driver.find_elements(By.CSS_SELECTOR, "p, li, h1, h2, h3")
            # Arrange the text and code in order
            indexed_elements = list(enumerate(all_elements))
            sorted_elements = sorted(indexed_elements, key=lambda x: x[0])
            self.history = [ele.text for idx, ele in sorted_elements]
            self.history.remove('ChatGPT')
        except:
            self.history = []

    def read_respond(self):
        l = []
        while len(l) == 0:
            try:
                all_elements = self.driver.find_elements(By.CSS_SELECTOR, "p, li, h1, h2, h3")
                indexed_elements = list(enumerate(all_elements))
                sorted_elements = sorted(indexed_elements, key=lambda x: x[0])
                for i in range(len(self.history), len(sorted_elements)-1):
                    response = sorted_elements[i][1].text
                    l.append(response)
                return l
            except:
                time.sleep(3)

    def new_chat(self):
        self.driver.find_elements("class name", 'text-token-text-primary')[3].click()

    def close(self):
        self.driver.quit()
        
def ask_gpt_for_final_answer(question, answer_one, answer_two):    
    driver = gptParser.get_driver()
    gpt_parser = gptParser(driver)

    query = "Here is a farmer question: " + question + " \\n " + "Answer 1: " + answer_one + " \\n Answer 2: " + answer_two + " \\n Which answer is more professional and focused? Please give each answer a separate score out of 100. Here is an example output: Answer1: {answer_1_score}\\n Answer2: {answer_2_score}\\nExplanation: "
    
    time.sleep(5)
    count = 0
    while driver.current_url != gpt_url:        
        if count > 10:
            print("Unable to connect to ChatGPT. Change IP....")
            driver.close()
            time.sleep(5)
            driver = gptParser.get_driver()
            gpt_parser = gptParser(driver)
            time.sleep(5)            
        else:
            count += 1
            print(f"Cannot open ChatGPT without login {count} times, Reopen a browser....")
            driver.close()
            time.sleep(5)
            driver = gptParser.get_driver()
            gpt_parser = gptParser(driver)
            time.sleep(5)
        

    time.sleep(5)
    gpt_parser(query)       
    
    time.sleep(20)    
    response = gpt_parser.read_respond()             
    
    time.sleep(5)
    driver.close()
    time.sleep(3)   
    
    return response   
    
def ask_gpt_about_question(question, num, fw):
    driver = gptParser.get_driver()
    gpt_parser = gptParser(driver)

    query = "There is a farmer asking a question which is : " + question + "  " + "Is this question an agricultural issue? Give me a short answer only includes True or False."
   
    time.sleep(5)
    gpt_parser(query)
    
    time.sleep(10)
    response = gpt_parser.read_respond()
    comfirm = ""
    
    for r in response:
        comfirm = comfirm + "\n" + r
        
    fw.write("Question comfirm {} :\n".format(num))
    fw.write(comfirm)
    fw.write("\n\n")
    
    time.sleep(10)
    driver.close()
    time.sleep(5)
    
    print(comfirm)
    print("==========")
    
    return
    

def ask_gpt_for_retrieve_result(question, answer_one, answer_two):
    driver = gptParser.get_driver()
    gpt_parser = gptParser(driver)
    
    query = "Here are a question and two contexts. \\n Question: " + question + " \\n " + "Context 1: " + answer_one + " \\n Context 2: " + answer_two + " \\n Give each context a relevant score from 0 to 100 based on its relation to the question. If a context is irrelevant to the question, the score must be lower than 60. Here is an example output: Context 1: 80\\nContext 2: 70\\nExplanation:"
    
    time.sleep(5)
    while driver.current_url != gpt_url:
        driver.close()
        time.sleep(5)
        driver = gptParser.get_driver()
        gpt_parser = gptParser(driver)
        time.sleep(5)

    time.sleep(5)
    gpt_parser(query)
    
    time.sleep(30)
    response = gpt_parser.read_respond()    
    
    time.sleep(10)
    driver.close()
    time.sleep(5)
    return response

def compare_identifiers_with_gpt(entry):       
    driver = gptParser.get_driver()
    gpt_parser = gptParser(driver)
    
    # Extract identifiers
    paper_parts = entry["paper"].split('\n')
    identifier_1 = [paper_parts[0], paper_parts[1] if len(paper_parts) > 1 else "", entry["section_title"]]
    identifier_2 = [paper_parts[0], paper_parts[1] if len(paper_parts) > 1 else "", entry["section_title_2"]]
    
    # Construct query
    query = f"""Here is a chunk for RAG implementation: \\n\\n Content: {entry['content']} \\n\\n I have two sets of identifiers (paper title part 1, paper title part 2, section title): \\n Identifier Set 1: {identifier_1} \\n Identifier Set 2: {identifier_2} \\n\\n Which set of identifiers is more appropriate and accurately represents the content? Please give each identifier set a separate score out of 100 based on relevance, accuracy, and how well it describes the content. \\n\\n Here is an example output format: \\nIdentifier Set 1: {{score}} \\nIdentifier Set 2: {{score}} \\nExplanation: {{your explanation}}"""
    
    time.sleep(5)
    count = 0
    
    while driver.current_url != gpt_url:        
        if count > 10:
            print("Unable to connect to ChatGPT. Change IP....")            
            driver.close()
            time.sleep(5)
            driver = gptParser.get_driver()
            gpt_parser = gptParser(driver)
            time.sleep(5)            
        else:
            count += 1
            print(f"Cannot open ChatGPT without login {count} times, Reopening browser....")
            driver.close()
            time.sleep(5)
            driver = gptParser.get_driver()
            gpt_parser = gptParser(driver)
            time.sleep(5)
        
    time.sleep(5)
    gpt_parser(query)
    
    time.sleep(20)
    response = gpt_parser.read_respond()
    
    time.sleep(5)
    driver.close()
    time.sleep(3)
    
    return response

input_dir = "input_file/"
output_dir = "output_file/"

if __name__ == "__main__":
    question_file = input_dir + "questions_100.txt"
    
    question_list = []
    
    with open(question_file, 'r') as qfr:
        for quest in qfr.read().split('\n'):
            question_list.append(quest)
        qfr.close()        
    
    file_1 = input_dir + "9907/9907_tart_stella1.5B_100Q_1st_Ans.txt"
    file_2 = input_dir + "9907/9907_tart_finetuned360k5ep_100Q_1st_Ans.txt"    
    
    answer_1_list = []
    answer_2_list = []    
    
    with open(file_1, 'r', encoding='utf-8') as fr1:
        for ans in fr1.read().split('Answer'):
            answer_1_list.append(ans)
        fr1.close()
        print(len(answer_1_list))
        answer_1_list.pop(30)
    
    with open(file_2, 'r') as fr2:
        for ans in fr2.read().split('Answer'):
            answer_2_list.append(ans)
        fr2.close()
        print(len(answer_2_list))
        print(answer_2_list[0])    
    
    output_dir = output_dir + "9907/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    file_out = output_dir + "stella_1.5B_llama_8B_360k2ep_FT_vs_tart_FT/"

    if not os.path.isdir(file_out):
        os.makedirs(file_out)
    
    files = os.listdir(file_out)

    numbers = []
    for file in files:
        if file.endswith('.txt') and file[:-4].isdigit():  # Check if it's a valid number
            numbers.append(int(file[:-4]))  # Extract the number part and convert to int

    largest_index = -1

    if numbers:
        largest_index = max(numbers)
        print(f"The largest index is: {largest_index}")
    else:
        print("No valid .txt files found.")

    init = largest_index + 1
    
    for i in range(init, len(question_list)):
        
        print(i)                
        # Normal RAG outputs
        answer_one = answer_1_list[i+1]
        answer_one = answer_one.split('{} :\n'.format(i))[1]
        answer_one = answer_one.replace('\n', ' ')
        answer_two = answer_2_list[i+1]
        answer_two = answer_two.split('{} :\n'.format(i))[1]
        answer_two = answer_two.replace('\n', ' ')        
        response = ask_gpt_for_final_answer(question_list[i], answer_one, answer_two)
        if(len(response)) < 3:
            sys.exit()
        with open(file_out + str(i) + '.txt', 'w') as fw:
            comparison = "\n".join(response)
            fw.write("Comparison {} :\n".format(i))
            fw.write(comparison)
            