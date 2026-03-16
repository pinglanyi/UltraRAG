import subprocess
import json

def get_all_datasets(base_url, api_key):
    all_datasets = []
    page = 1
    page_size = 100  # 每页最多取100条

    while True:
        result = subprocess.run(
            [
                "curl", "-s",
                f"{base_url}/api/v1/datasets?page={page}&page_size={page_size}",
                "-H", f"Authorization: Bearer {api_key}"
            ],
            capture_output=True, text=True
        )

        data = json.loads(result.stdout)
        datasets = data.get("data", [])

        if not datasets:
            break

        all_datasets.extend(datasets)
        print(f"第 {page} 页，获取 {len(datasets)} 条，累计 {len(all_datasets)} 条")

        # 不足一页说明已是最后一页
        if len(datasets) < page_size:
            break

        page += 1

    return all_datasets

# ── 配置 ──────────────────────────────────────────────────
BASE_URL = "http://10.65.8.100:9222"
API_KEY  = "ragflow-pFtiRyvxRi9TKTNkBdbgm8DTwmyvhmNRztdeWD9dfFM"
OUTPUT   = "servers/ragflow_retriever/parameter.yaml"

# ── 获取全部 datasets ─────────────────────────────────────
import os
os.makedirs("servers/ragflow_retriever", exist_ok=True)

datasets = get_all_datasets(BASE_URL, API_KEY)
print(f"\n共获取 {len(datasets)} 个知识库")

# ── 写入 parameter.yaml ───────────────────────────────────
with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("# RAGFlow 连接\n")
    f.write(f'ragflow_base_url: "{BASE_URL}"\n')
    f.write(f'ragflow_api_key: "{API_KEY}"\n\n')
    f.write("# 知识库\n")
    f.write("ragflow_dataset_ids:\n")
    for d in datasets:
        f.write(f'  - "{d["id"]}"    # {d.get("name", "")}\n')
    f.write("\n# 检索参数\n")
    f.write("top_k: 5\n")
    f.write("similarity_threshold: 0.2\n")
    f.write("keywords_similarity_weight: 0.7\n")
    f.write("max_doc_len: 2000\n")

print(f"已写入 {OUTPUT}")