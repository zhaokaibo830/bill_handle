import os
import base64
import shutil
import logging
import uuid
import tempfile
from pathlib import Path
from typing import List

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from tqdm import tqdm

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI, APIError, APITimeoutError

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- 配置 ---
# 建议将 API KEY 放入环境变量中: export DASHSCOPE_API_KEY="sk-xxx"
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-7cc21f360b1342ceb3afb3fe1a3ce5c8")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 初始化 Client (建议在启动时检查 Key)
if not API_KEY or API_KEY.startswith("sk-xxx"):
    logger.warning("API Key 未配置或无效，请检查环境变量或代码配置。")

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    logger.error(f"OpenAI Client 初始化失败: {e}")
    client = None

app = FastAPI()

# 确保结果目录存在
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def encode_image(image_path: str) -> str:
    """将图片文件转换为 Base64 编码字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"读取图片失败 {image_path}: {e}")
        raise HTTPException(status_code=500, detail=f"内部错误: 无法处理图片文件")


def load_images_from_pdf(pdf_path: str, output_folder: str, dpi: int = 200) -> List[str]:
    """
    将 PDF 转换为图片并保存到指定文件夹。
    返回生成的图片路径列表。
    """
    image_paths = []
    try:
        # 使用 fitz.open 的上下文管理器，确保文件关闭
        with fitz.open(pdf_path) as doc:
            if doc.page_count == 0:
                raise ValueError("PDF 文件为空")

            for index, page in enumerate(doc):
                # 限制最大分辨率以防止内存溢出
                matrix = fitz.Matrix(dpi / 72, dpi / 72)
                pm = page.get_pixmap(matrix=matrix, alpha=False)

                # 如果图片过大，降低分辨率重试
                if pm.width > 9000 or pm.height > 9000:
                    logger.warning(f"第 {index} 页尺寸过大，正在降级分辨率处理...")
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                # 转换颜色空间
                img_data = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, 3)  # RGB
                # OpenCV 默认是 BGR，如果需要用 OpenCV 保存，通常需要转 BGR，或者直接用 PIL 保存
                # 这里为了兼容原逻辑使用 OpenCV 保存
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

                img_path = os.path.join(output_folder, f"{index}.png")
                cv2.imwrite(img_path, img_bgr)
                image_paths.append(img_path)

    except fitz.FileDataError:
        raise HTTPException(status_code=400, detail="文件损坏或不是有效的 PDF 文件")
    except Exception as e:
        logger.error(f"PDF 转图片失败: {e}")
        raise HTTPException(status_code=500, detail=f"PDF 处理失败: {str(e)}")

    return image_paths


def call_llm_processing(current_img_path: str, reference_img_path: str = None) -> str:
    """调用大模型进行图片分析"""
    if not client:
        raise HTTPException(status_code=500, detail="大模型客户端未初始化")

    base64_current = encode_image(current_img_path)

    messages = []

    if reference_img_path is None:
        # 第一页的处理逻辑
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_current}"}},
            {"type": "text",
             "text": "请将下方流水图片转换为Markdown格式，并使用规范的Markdown表格语法呈现其中的表格数据。请确保表格的行列对应关系准确无误，且不遗漏任何数据。"}
        ]
    else:
        # 后续页的处理逻辑（参考第一页）
        base64_ref = encode_image(reference_img_path)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_ref}"}},  # 参考图
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_current}"}},  # 当前图
            {"type": "text",
             "text": "请参考第一张流水图中的内容，将第二张流水图片转换为 Markdown 格式，并使用规范的 Markdown 表格语法准确呈现其中的表格数据。请确保表格的列与行对应关系正确，且不遗漏任何信息。"}
        ]

    messages = [{"role": "user", "content": content}]

    try:
        completion = client.chat.completions.create(
            model="qwen3-vl-32b-instruct",
            messages=messages,
            timeout=120  # 设置超时时间，防止无限等待
        )
        return completion.choices[0].message.content
    except APITimeoutError:
        logger.error("LLM 请求超时")
        return "\n\n> **错误**: 该页面处理超时。\n\n"
    except APIError as e:
        logger.error(f"LLM API 错误: {e}")
        return f"\n\n> **错误**: 模型服务异常 ({e.code})。\n\n"
    except Exception as e:
        logger.error(f"LLM 未知错误: {e}")
        return "\n\n> **错误**: 处理过程中发生未知错误。\n\n"


@app.post("/api/bill2text")
async def bill2text(file: UploadFile = File(...)):
    # 1. 验证文件类型
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    request_id = str(uuid.uuid4())
    logger.info(f"收到请求 ID: {request_id}, 文件名: {file.filename}")

    # 使用 TemporaryDirectory 创建沙盒环境
    # with 语句结束时，temp_dir 会自动被删除，解决资源残留和并发冲突问题
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_save_path = temp_path / f"input_{request_id}.pdf"
        img_output_folder = temp_path / "imgs"
        img_output_folder.mkdir()

        # 2. 保存上传的文件
        try:
            with open(pdf_save_path, "wb") as f:
                # 使用 shutil.copyfileobj 处理大文件比 read() 更高效
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            logger.error(f"文件保存失败: {e}")
            raise HTTPException(status_code=500, detail="文件上传失败")
        finally:
            file.file.close()  # 显式关闭 UploadFile

        # 3. PDF 转图片
        logger.info(f"[{request_id}] 开始转换 PDF 为图片...")
        image_paths = load_images_from_pdf(str(pdf_save_path), str(img_output_folder))

        if not image_paths:
            raise HTTPException(status_code=400, detail="PDF 解析未生成任何图片")

        logger.info(f"[{request_id}] 转换完成，共 {len(image_paths)} 页。开始调用 LLM...")

        # 4. 循环处理每一页
        full_text = ""
        first_page_path = image_paths[0]  # 保存第一页路径作为参考

        # 使用 tqdm 显示进度 (注意：tqdm 在 Web 服务日志中可能会导致混乱，这里仅作保留)
        for i, img_path in enumerate(image_paths):
            logger.info(f"[{request_id}] 正在处理第 {i + 1}/{len(image_paths)} 页")

            if i == 0:
                page_text = call_llm_processing(img_path, reference_img_path=None)
            else:
                page_text = call_llm_processing(img_path, reference_img_path=first_page_path)

            full_text += f"\n\n## Page {i + 1}\n\n" + page_text

        # 5. 保存结果文件 (如果不希望保存到本地，可以去掉这一步)
        # 使用安全的文件名，防止路径遍历攻击
        safe_filename = Path(file.filename).stem + f"_{request_id[:8]}.md"
        result_file_path = RESULTS_DIR / safe_filename

        try:
            with open(result_file_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            logger.info(f"[{request_id}] 处理完成，结果已保存至 {result_file_path}")
        except Exception as e:
            logger.error(f"[{request_id}] 结果文件写入失败: {e}")
            # 即使文件写入失败，也可以尝试返回文本给前端

        return {
            "status": "success",
            "filename": safe_filename,
            "text": full_text
        }


if __name__ == "__main__":
    import uvicorn

    # 建议生产环境 worker 数量 > 1
    uvicorn.run(app, host="0.0.0.0", port=6006)