"""测试素材生成：确定性绘制文本/UI 图片（macOS 优先；Linux 上仅英文可用，DejaVu 无 CJK 字形）"""
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    # macOS 15 中 PingFang 已迁移至 FontServices 私有路径，不再使用稳定路径
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(size: int = 32):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise RuntimeError("no usable font found")


def make_text_image(path, title="Invoice #2024-0715 Total: $1,234.56",
                    body="你好，世界 这是一张测试发票", size=(800, 300)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.text((40, 40), title, font=find_font(36), fill="black")
    d.text((40, 140), body, font=find_font(32), fill="black")
    img.save(path)
    return str(path)


def make_ui_image(path, size=(640, 480)):
    """深色背景 UI：#1F2937 底；标题 #6B7280（低对比度，应被检出）；两按钮白色。"""
    img = Image.new("RGB", size, (31, 41, 55))
    d = ImageDraw.Draw(img)
    d.text((40, 40), "商品详情", font=find_font(40), fill=(107, 114, 128))
    d.text((40, 160), "加入购物车", font=find_font(24), fill="white")
    d.text((40, 220), "立即购买", font=find_font(24), fill="white")
    img.save(path)
    return str(path)
