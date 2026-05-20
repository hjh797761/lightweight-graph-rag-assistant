from flask import Flask, request, jsonify
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphrag_assistant import retrieve, generate_answer, load_knowledge_base

app = Flask(__name__)

DINGTALK_WEBHOOK = os.getenv("DINGTALK_WEBHOOK", "")

def send_to_dingtalk(text):
    if not DINGTALK_WEBHOOK:
        raise RuntimeError("请先设置环境变量 DINGTALK_WEBHOOK。")
    data = {
        "msgtype": "text",
        "text": {"content": text}
    }
    requests.post(DINGTALK_WEBHOOK, json=data)

@app.route("/dingtalk", methods=["POST"])
def dingtalk():
    body = request.json or {}
    query = body.get("text", {}).get("content", "").strip()
    
    if not query:
        return jsonify({"success": True})
    
    try:
        chapter, sub, context = retrieve(query)
        answer = generate_answer(query, context)
        
        # 检索路径
        path = f"{chapter} → {sub}" if sub else chapter
        
        # 提取概念关系部分
        relations_text = ""
        if context and "【关系】" in context:
            relations_part = context.split("【关系】")[1].strip()
            if relations_part:
                relations_text = f"\n\n【概念关系】\n{relations_part}"
        
        # 调试：附上检索原文片段（验证是否真实检索）
        debug_context = ""
        if context:
            raw = context.split("【关系】")[0].strip()
            debug_context = f"\n\n【检索原文片段】\n{raw[:200]}..."
        
        reply = f"【检索路径】\n{path}{relations_text}{debug_context}\n\n【回答】\n{answer}"
        send_to_dingtalk(reply)
        
    except Exception as e:
        send_to_dingtalk(f"出错了：{e}")
    
    return jsonify({"success": True})

if __name__ == "__main__":
    load_knowledge_base()
    print("服务启动中...")
    app.run(host="0.0.0.0", port=5000)
