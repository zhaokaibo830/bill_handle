from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from openai import OpenAI
import base64
import shutil
import os
import fitz
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm,trange

app = FastAPI()
#  编码函数： 将本地文件转换为 Base64 编码的字符串


def load_images_from_pdf(pdf_path, img_folder, dpi=200):
    os.makedirs(img_folder, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            pm = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)

            if pm.width > 9000 or pm.height > 9000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = np.array(Image.frombytes("RGB", (pm.width, pm.height), pm.samples))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            img_path = os.path.join(img_folder, f"{index}.png")
            cv2.imwrite(img_path, img)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

client = OpenAI(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
                # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
                api_key="sk-7cc21f360b1342ceb3afb3fe1a3ce5c8",
                # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

@app.post("/api/bill2text")
def bill2text(file: UploadFile = File(...)):
    try:
        file_name = file.filename
        save_path = "./uploaded_files"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_filepath = os.path.join(save_path, file_name)
        with open(save_filepath, "wb") as f:
            f.write(file.file.read())
    except AttributeError:
        return JSONResponse(content={"error": "文件上传出错"})
    shutil.rmtree("./imgs")
    os.makedirs("./imgs", exist_ok=True)
    load_images_from_pdf(save_filepath, img_folder=f"./imgs")
    page_num=len(os.listdir("./imgs"))
    text=""
    for i in trange(page_num):
        # print(f"------------正在处理第{i}页-----------")
        if i==0:
            base64_image = encode_image(rf"./imgs/{i}.png")
            completion = client.chat.completions.create(
                model="qwen3-vl-32b-instruct",
                # 此处以qwen3-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                                # PNG图像：  f"data:image/png;base64,{base64_image}"
                                # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                                # WEBP图像： f"data:image/webp;base64,{base64_image}"
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                            {"type": "text",
                             "text": "请将下方流水图片转换为Markdown格式，并使用规范的Markdown表格语法呈现其中的表格数据。请确保表格的行列对应关系准确无误，且不遗漏任何数据。"},
                        ],
                    }
                ],
            )
            text=completion.choices[0].message.content
        else:
            # 将xxxx/eagle.png替换为你本地图像的绝对路径
            base64_image1 = encode_image(r"./imgs/0.png")

            base64_image2 = encode_image(rf"./imgs/{i}.png")
            completion = client.chat.completions.create(
                model="qwen3-vl-32b-instruct", # 此处以qwen3-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                                # PNG图像：  f"data:image/png;base64,{base64_image}"
                                # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                                # WEBP图像： f"data:image/webp;base64,{base64_image}"
                                "image_url": {"url": f"data:image/png;base64,{base64_image1}"},
                            },
            {
                                "type": "image_url",
                                # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                                # PNG图像：  f"data:image/png;base64,{base64_image}"
                                # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                                # WEBP图像： f"data:image/webp;base64,{base64_image}"
                                "image_url": {"url": f"data:image/png;base64,{base64_image2}"},
                            },
                            {"type": "text", "text": "请参考第一张流水图中的内容，将第二张流水图片转换为 Markdown 格式，并使用规范的 Markdown 表格语法准确呈现其中的表格数据。请确保表格的列与行对应关系正确，且不遗漏任何信息。"},
                        ],
                    }
                ],
            )
            text+=completion.choices[0].message.content
        # 打开文件进行写入（如果文件不存在则创建，存在则覆盖）
        with open(f'./results/{file_name}.md', 'w', encoding='utf-8') as file:
            file.write(text)
    with open(f'./results/{file_name}.md', 'w', encoding='utf-8') as file:
        file.write(text)
    return {"text":text}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)