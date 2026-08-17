"""集成测试：真实调用 PP-OCRv6（需先跑过 --smoke 完成模型下载）"""
import pytest

from image_factory import make_text_image, make_ui_image
from ocr import build_engine, ocr_file, sort_lines_reading_order, warmup

ENGINE = None


def get_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = build_engine("medium")
        warmup(ENGINE)
    return ENGINE


def _texts(results):
    return [ln["text"] for r in results for ln in r["lines"]]


@pytest.mark.integration
def test_english_invoice_text_mode(tmp_path):
    p = make_text_image(tmp_path / "invoice.png")
    results = ocr_file(get_engine(), p)
    texts = _texts(results)
    joined = " ".join(texts)
    assert "Invoice" in joined
    assert "2024" in joined
    assert "1234" in joined.replace(",", "").replace(" ", "") or "1,234" in joined


@pytest.mark.integration
def test_chinese_and_reading_order(tmp_path):
    p = make_text_image(tmp_path / "cn.png", title="第一行标题", body="第二行正文内容")
    results = ocr_file(get_engine(), p)
    ordered = sort_lines_reading_order(
        [ln for r in results for ln in r["lines"]]
    )
    assert len(ordered) >= 2
    assert "第一行" in ordered[0]["text"]
    assert "第二行" in ordered[1]["text"]
    joined = " ".join(_texts(results))
    assert "第二行" in joined or "正文" in joined


@pytest.mark.integration
def test_json_mode_has_coords(tmp_path):
    p = make_text_image(tmp_path / "j.png")
    results = ocr_file(get_engine(), p)
    assert len(results) == 1
    r = results[0]
    assert r["width"] > 0 and r["height"] > 0
    assert r["lines"]
    for ln in r["lines"]:
        assert len(ln["box"]) == 4
        assert ln["box"][0] < ln["box"][2] and ln["box"][1] < ln["box"][3]
        assert ln["font_size"] > 0
        assert isinstance(ln["conf"], float)
    ys = [ln["box"][1] for ln in r["lines"]]
    assert ys == sorted(ys)


@pytest.mark.integration
def test_pdf_multi_page(tmp_path):
    import pymupdf as fitz
    from ocr import load_images
    pdf = tmp_path / "two.pdf"
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Page Content", fontsize=24)
    doc.save(pdf)
    doc.close()
    pages = list(load_images(str(pdf)))
    assert len(pages) == 2
    assert pages[0][1:] == (1, 2) and pages[1][1:] == (2, 2)
    results = ocr_file(get_engine(), str(pdf))
    assert len(results) == 2
    assert any("Page" in ln["text"] for r in results for ln in r["lines"])


@pytest.mark.integration
def test_table_mode(tmp_path):
    from image_factory import find_font
    from ocr import build_tsv
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (600, 200), "white")
    d = ImageDraw.Draw(img)
    f = find_font(24)
    d.text((20, 20), "名称", font=f, fill="black")
    d.text((250, 20), "数量", font=f, fill="black")
    d.text((400, 20), "价格", font=f, fill="black")
    d.text((20, 80), "苹果", font=f, fill="black")
    d.text((250, 80), "3", font=f, fill="black")
    d.text((400, 80), "9.90", font=f, fill="black")
    p = tmp_path / "tbl.png"
    img.save(p)
    results = ocr_file(get_engine(), str(p))
    tsv = build_tsv([ln for r in results for ln in r["lines"]])
    assert "名称\t数量\t价格" in tsv
    assert "苹果\t3\t9.90" in tsv
    assert "苹果" in tsv
    assert "9.90" in tsv
    assert len(tsv.splitlines()) >= 2
