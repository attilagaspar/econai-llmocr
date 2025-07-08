import sys
import os
import json
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QPoint
from PIL import Image

LABEL_COLORS = {
    "text_cell": QColor(255, 0, 0, 120),        # Red
    "numerical_cell": QColor(0, 200, 0, 120),   # Green
    "column_header": QColor(0, 0, 255, 120),    # Blue
}

class ImageWithBoxes(QLabel):
    boxClicked = pyqtSignal(int)  # index in shapes array

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.shapes = []
        self.selected_idx = None

    def load(self, image_path, shapes):
        self.image = QPixmap(image_path)
        self.shapes = shapes
        self.selected_idx = None
        self.repaint()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image:
            painter = QPainter(self)
            scaled = self.image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, scaled)
            scale_x = scaled.width() / self.image.width()
            scale_y = scaled.height() / self.image.height()
            offset_x = (self.width() - scaled.width()) // 2
            offset_y = (self.height() - scaled.height()) // 2
            for idx, shape in enumerate(self.shapes):
                if "points" not in shape or len(shape["points"]) < 2:
                    continue
                x1, y1 = shape["points"][0]
                x2, y2 = shape["points"][1]
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                label = shape.get("label", "")
                color = LABEL_COLORS.get(label, QColor(128, 128, 128, 120))
                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                rect = QRect(
                    int(offset_x + x1 * scale_x),
                    int(offset_y + y1 * scale_y),
                    int((x2 - x1) * scale_x),
                    int((y2 - y1) * scale_y)
                )
                painter.drawRect(rect)
                if idx == self.selected_idx:
                    painter.setPen(QPen(Qt.yellow, 3, Qt.DashLine))
                    painter.drawRect(rect)
            painter.end()

    def mousePressEvent(self, event):
        if not self.image or not self.shapes:
            return
        scaled = self.image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scale_x = scaled.width() / self.image.width()
        scale_y = scaled.height() / self.image.height()
        offset_x = (self.width() - scaled.width()) // 2
        offset_y = (self.height() - scaled.height()) // 2
        x = (event.x() - offset_x) / scale_x
        y = (event.y() - offset_y) / scale_y
        for idx, shape in enumerate(self.shapes):
            if "points" not in shape or len(shape["points"]) < 2:
                continue
            x1, y1 = shape["points"][0]
            x2, y2 = shape["points"][1]
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_idx = idx
                self.repaint()
                self.boxClicked.emit(idx)
                break

class MainWindow(QWidget):
    def __init__(self, input_dir):
        super().__init__()
        self.setWindowTitle("OCR/LLM/Human Table Correction Tool")
        self.input_dir = input_dir
        self.json_files = self._find_jsons()
        self.current_idx = 0
        self.current_shape_idx = None
        self.timer_start = None

        # Left: image with boxes
        self.image_label = ImageWithBoxes()
        self.image_label.setMinimumWidth(800)
        self.image_label.setMinimumHeight(1000)
        self.image_label.boxClicked.connect(self.on_box_clicked)

        # Right: 3 text boxes and buttons
        self.ocr_box = QTextEdit()
        self.ocr_box.setReadOnly(True)
        self.ocr_btn = QPushButton("Choose")
        self.ocr_btn.clicked.connect(self.choose_ocr)

        self.llm_box = QTextEdit()
        self.llm_box.setReadOnly(True)
        self.llm_btn = QPushButton("Choose")
        self.llm_btn.clicked.connect(self.choose_llm)

        self.human_box = QTextEdit()
        self.human_btn = QPushButton("Save")
        self.human_btn.clicked.connect(self.save_human)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("OCR output:"))
        right_layout.addWidget(self.ocr_box)
        right_layout.addWidget(self.ocr_btn)
        right_layout.addWidget(QLabel("LLM output:"))
        right_layout.addWidget(self.llm_box)
        right_layout.addWidget(self.llm_btn)
        right_layout.addWidget(QLabel("Human correction:"))
        right_layout.addWidget(self.human_box)
        right_layout.addWidget(self.human_btn)

        layout = QHBoxLayout()
        layout.addWidget(self.image_label, 2)
        layout.addLayout(right_layout, 1)
        self.setLayout(layout)

        self.load_page(self.current_idx)

    def _find_jsons(self):
        jsons = []
        for root, _, files in os.walk(self.input_dir):
            for f in sorted(files):
                if f.lower().endswith(".json"):
                    jsons.append(os.path.join(root, f))
        return jsons

    def load_page(self, idx):
        if not (0 <= idx < len(self.json_files)):
            return
        self.current_idx = idx
        self.current_shape_idx = None
        self.timer_start = None
        json_path = self.json_files[idx]
        img_path = os.path.splitext(json_path)[0]
        for ext in [".jpg", ".jpeg", ".png"]:
            if os.path.exists(img_path + ext):
                img_path = img_path + ext
                break
        else:
            self.image_label.clear()
            return
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.image_label.load(img_path, self.data.get("shapes", []))
        self.ocr_box.clear()
        self.llm_box.clear()
        self.human_box.clear()
        self.setWindowTitle(f"OCR/LLM/Human Table Correction Tool - {os.path.basename(json_path)}")

    def on_box_clicked(self, idx):
        self.current_shape_idx = idx
        self.timer_start = time.time()
        shape = self.data["shapes"][idx]
        # Load OCR
        ocr_text = ""
        if "tesseract_output" in shape and "ocr_text" in shape["tesseract_output"]:
            ocr_text = shape["tesseract_output"]["ocr_text"]
        self.ocr_box.setPlainText(ocr_text)
        # Load LLM
        llm_text = ""
        if "openai_output" in shape and "response" in shape["openai_output"]:
            llm_text = shape["openai_output"]["response"]
        self.llm_box.setPlainText(llm_text)
        # Load human
        human_text = ""
        if "human_output" in shape and "human_corrected_text" in shape["human_output"]:
            human_text = shape["human_output"]["human_corrected_text"]
        self.human_box.setPlainText(human_text)

    def choose_ocr(self):
        self.human_box.setPlainText(self.ocr_box.toPlainText())

    def choose_llm(self):
        self.human_box.setPlainText(self.llm_box.toPlainText())

    def save_human(self):
        if self.current_shape_idx is None:
            return
        shape = self.data["shapes"][self.current_shape_idx]
        elapsed = None
        if self.timer_start:
            elapsed = time.time() - self.timer_start
        else:
            elapsed = 0
        if "human_output" not in shape:
            shape["human_output"] = {}
        shape["human_output"]["human_corrected_text"] = self.human_box.toPlainText()
        shape["human_output"]["human_processing_time"] = elapsed
        # Save JSON
        with open(self.json_files[self.current_idx], "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        self.timer_start = time.time()

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.AltModifier and event.key() == Qt.Key_N:
            self.load_page(self.current_idx + 1)
        elif event.modifiers() == Qt.AltModifier and event.key() == Qt.Key_P:
            self.load_page(self.current_idx - 1)
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_llm_human_gui.py <input_dir>")
        sys.exit(1)
    app = QApplication(sys.argv)
    window = MainWindow(sys.argv[1])
    window.show()
    sys.exit(app.exec_())