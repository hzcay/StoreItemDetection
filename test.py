from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    use_gpu=False
)

res = ocr.predict("D:/CV/StoreItemDetection/uploads/0aa0d6c9-b03a-4e39-a571-a17044c63e58.jpg")
print(res)
