import os
import re
import json
import random
import requests
import bz2
from tqdm import tqdm
import sys

# --------------------------
# 配置
# --------------------------
WIKI_URL = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
WIKI_BZ2 = "zhwiki-latest-pages-articles.xml.bz2"
WIKIX_DIR = "wikiextracted"
CORPUS_FILE = "corpus.txt"
TARGET_LINES = 500000  # 最终 corpus.txt 行数

# --------------------------
# 下载中文维基 XML（如果不存在）
# --------------------------
if not os.path.exists(WIKI_BZ2):
    print(f"🌐 开始下载中文维基百科 XML 数据（约 1~2 GB）...")
    with requests.get(WIKI_URL, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(WIKI_BZ2, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True, desc=WIKI_BZ2
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    print("✅ 下载完成")
else:
    print(f"✅ 已存在 {WIKI_BZ2}，跳过下载")

# --------------------------
# 检查 WikiExtractor 是否安装
# --------------------------
try:
    import wikiextractor
except ImportError:
    print("⚠️ WikiExtractor 未安装，开始自动安装...")
    os.system(f"{sys.executable} -m pip install wikiextractor")

# --------------------------
# 使用 WikiExtractor 提取文本（仅生成少量 JSON）
# --------------------------
if not os.path.exists(WIKIX_DIR):
    os.makedirs(WIKIX_DIR, exist_ok=True)

print("🌐 开始提取中文维基文本（小型化，仅前几千条）...")
# 使用 Python API 流式处理 bz2 文件
def extract_wiki_small(bz2_file, output_dir, max_articles=10000):
    import xml.etree.ElementTree as ET

    def clean_text(text):
        return re.sub(r'\s+', ' ', text).strip()

    count = 0
    with bz2.open(bz2_file, "rt", encoding="utf-8", errors="ignore") as f:
        article_lines = []
        in_page = False
        for line in tqdm(f):
            if "<page>" in line:
                in_page = True
                article_lines = [line]
            elif "</page>" in line:
                article_lines.append(line)
                in_page = False
                xml_str = "".join(article_lines)
                try:
                    root = ET.fromstring(xml_str)
                    title = root.find('title').text
                    text_node = root.find('.//revision/text')
                    if text_node is None or text_node.text is None:
                        continue
                    text = clean_text(text_node.text)
                    # 写入 JSON
                    file_idx = count // 1000
                    os.makedirs(os.path.join(output_dir, f"{file_idx:03d}"), exist_ok=True)
                    out_path = os.path.join(output_dir, f"{file_idx:03d}", f"{file_idx:03d}_{count%1000:04d}.json")
                    with open(out_path, "w", encoding="utf-8") as fout:
                        json.dump({"title": title, "text": text}, fout, ensure_ascii=False)
                    count += 1
                    if count >= max_articles:
                        return
                except Exception:
                    continue
            elif in_page:
                article_lines.append(line)

extract_wiki_small(WIKI_BZ2, WIKIX_DIR, max_articles=10000)
print(f"✅ 提取完成，约 {10000} 条文章保存到 {WIKIX_DIR}")

# --------------------------
# 生成 corpus.txt
# --------------------------
print(f"🌐 开始生成 {CORPUS_FILE} ...")

all_sentences = []
for root, dirs, files in os.walk(WIKIX_DIR):
    for file in files:
        if not file.endswith(".json"):
            continue
        path = os.path.join(root, file)
        with open(path, "r", encoding="utf-8") as fin:
            data = json.load(fin)
            text = data.get("text", "")
            sentences = re.split(r"[。！？]", text)
            for sent in sentences:
                sent = sent.strip()
                if 10 < len(sent) < 200:  # 过滤短句子
                    all_sentences.append(sent)

# 随机抽取 TARGET_LINES 条
if len(all_sentences) > TARGET_LINES:
    selected_sentences = random.sample(all_sentences, TARGET_LINES)
else:
    selected_sentences = all_sentences

with open(CORPUS_FILE, "w", encoding="utf-8") as fout:
    for sent in selected_sentences:
        fout.write(sent + "\n")

print(f"✅ 已生成 {CORPUS_FILE}，共 {len(selected_sentences)} 条语料")
