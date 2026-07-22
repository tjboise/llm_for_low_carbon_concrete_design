import google.generativeai as genai
import os
import time

# ==========================================
# 1. 配置 API Key 和模型
# ==========================================
# 已根据你的要求直接配置 API Key
genai.configure(api_key="AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo")

# 使用你指定的 gemini-2.0-flash 模型
# 注意：2.0 Flash 速度极快，非常适合这种多文档处理
model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 2. 设置 PDF 文件夹路径
# ==========================================
# 请确保你的 PDF 文件放在当前脚本同级目录下的 'my_drawings' 文件夹里
# 或者修改下面的路径为你的实际存放位置
pdf_folder_path = "./my_drawings"

if not os.path.exists(pdf_folder_path):
    os.makedirs(pdf_folder_path)
    print(f"已为你创建文件夹 '{pdf_folder_path}'，请把 PDF 图纸放进去后重新运行程序。")
    exit()

pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

if not pdf_files:
    print(f"错误：在 '{pdf_folder_path}' 文件夹内没找到 PDF 文件。")
    exit()

# ==========================================
# 3. 上传文件到 Gemini File API
# ==========================================
uploaded_files = []
print(f"正在准备上传 {len(pdf_files)} 个施工文档...")

for file_path in pdf_files:
    display_name = os.path.basename(file_path)
    print(f"正在上传并处理: {display_name}...", end="", flush=True)
    try:
        # 上传大文件
        myfile = genai.upload_file(path=file_path, display_name=display_name)

        # 等待服务器处理完成（Active状态）
        while myfile.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(3)
            myfile = genai.get_file(myfile.name)

        if myfile.state.name == "FAILED":
            print(f" 失败！")
            continue

        uploaded_files.append(myfile)
        print(f" 完成！")
    except Exception as e:
        print(f" 出错: {e}")

if not uploaded_files:
    print("没有文件上传成功。")
    exit()

# ==========================================
# 4. 开启对话模式并生成初始方案
# ==========================================

print("\n" + "=" * 50)
print("正在阅读图纸并编制方案初稿，请稍候...")
print("=" * 50 + "\n")

# 构造初始 Prompt
initial_prompt = """
你现在是一位资深路桥工程总工程师。我是一名刚从房建转行到路桥的施工员。
我已经上传了本项目的所有 PDF 施工图纸和交通组织方案。

请结合这些文档中的具体数据（桩号、工程量、技术要求），为我编制以下四个施工方案的草案：
1. 临时交通道路施工方案 (重点参考交改方案文档)
2. 排水工程施工方案 (重点参考管线图纸)
3. 桥梁桩基施工方案 (重点参考桥梁结构图)
4. 支座垫石施工方案 (请先科普概念，再写出严苛的标高控制流程)

要求：方案要专业、具体，直接引用图纸中的参数，不要只给模版。
"""

# 启动对话 Session
chat_session = model.start_chat(history=[])

# 发送文件和初始指令
try:
    # 将文件对象列表和文字合并发送
    contents = uploaded_files + [initial_prompt]
    response = chat_session.send_message(contents, stream=True)

    print("--- Gemini 生成的初始方案 ---\n")
    for chunk in response:
        print(chunk.text, end="", flush=True)
    print("\n" + "-" * 30)

except Exception as e:
    print(f"生成方案时出错: {e}")
    exit()

# ==========================================
# 5. 进入持续对话阶段
# ==========================================
print("\n[系统消息]：初稿已完成。作为一个新手，你可以针对任何不懂的地方继续问我。")
print("例如输入：'详细解释一下垫石怎么测量标高' 或 '桩基方案里用的什么泥浆指标'。")
print("输入 'exit' 退出程序。")

while True:
    user_query = input("\n你的问题 >>> ")

    if user_query.lower() in ['exit', 'quit', '退出']:
        print("祝你施工顺利，注意安全！再见。")
        break

    try:
        response = chat_session.send_message(user_query, stream=True)
        print("\nGemini 总工回复：\n")
        for chunk in response:
            print(chunk.text, end="", flush=True)
        print("\n" + "-" * 20)
    except Exception as e:
        print(f"对话发生错误: {e}")