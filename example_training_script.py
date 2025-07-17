from efficient_ocr import EffOCR


effocr = EffOCR(
    '../traindata_ocr/config_proba.yaml',
    '../traindata_ocr/labels.json',
    '../traindata_ocr/images',
   )
effocr.train(target=['char_recognizer'])
