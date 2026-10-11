from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu:0",
)
print("Models loaded")

result = ocr.predict("input/test.png")  # replace with any image that has text
for r in result:
    print(list(zip(r["rec_texts"], r["rec_scores"])))
