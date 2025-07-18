import sys
import os
import json
import time
import re
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog, QPlainTextEdit
)
from PyQt5.QtWidgets import QTextEdit, QWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QRect, QSize
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QTextFormat
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QImage, QTextFormat
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QPoint
from PIL import Image

LABEL_COLORS = {
    "text_cell": QColor(255, 0, 0, 120),        # Red
    "numerical_cell": QColor(0, 200, 0, 120),   # Green
    "column_header": QColor(0, 0, 255, 120),    # Blue
}

def natural_key(s):
    # Split string into list of strings and integers: "page_11.json" -> ["page_", 11, ".json"]
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def pil2pixmap(im):
    #if im.mode != "RGB":
    #    im = im.convert("RGB")
    #data = im.tobytes("raw", "RGB")
    #qimg = QImage(data, im.size[0], im.size[1], QImage.Format_RGB888)
    try: 
        from PIL.ImageQt import ImageQt
        return QPixmap.fromImage(ImageQt(im))
    except ImportError:
        # Fallback if ImageQt is not available
        if im.mode != "RGB":
            im = im.convert("RGB")
        data = im.tobytes("raw", "RGB")
        w, h = im.size
        bytes_per_line = w * 3
        qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)


class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint_event(event)

class LineNumberedTextEdit(QPlainTextEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lineNumberArea = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.highlight_current_line)
        self.update_line_number_area_width(0)
        self.highlight_current_line()

    def line_number_area_width(self):
        digits = len(str(max(1, self.blockCount())))
        space = 3 + self.fontMetrics().width('9') * digits
        return space

    def update_line_number_area_width(self, _):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        if dy:
            self.lineNumberArea.scroll(0, dy)
        else:
            self.lineNumberArea.update(0, rect.y(), self.lineNumberArea.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.lineNumberArea.setGeometry(QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height()))

    def line_number_area_paint_event(self, event):
        painter = QPainter(self.lineNumberArea)
        painter.fillRect(event.rect(), Qt.lightGray)
        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        height = self.fontMetrics().height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(blockNumber + 1)
                painter.setPen(Qt.gray)
                painter.drawText(0, top, self.lineNumberArea.width() - 4, height,
                                 Qt.AlignRight, number)
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            blockNumber += 1

    def highlight_current_line(self):
        extraSelections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            lineColor = QColor(Qt.yellow).lighter(160)
            selection.format.setBackground(lineColor)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)            
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extraSelections.append(selection)
        self.setExtraSelections(extraSelections)


class ImageWithBoxes(QLabel):
    boxClicked = pyqtSignal(int)  # index in shapes array

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.shapes = []
        self.selected_idx = None
        self.scaled_size = None
        self.offset_x = 0
        self.offset_y = 0

    def load(self, image_path, shapes):
        self.image = QPixmap(image_path)
        self.shapes = shapes
        self.selected_idx = None
        self.update_scaled_params()
        self.repaint()

    def update_scaled_params(self):
        if not self.image:
            self.scaled_size = None
            self.offset_x = 0
            self.offset_y = 0
            return
        widget_w, widget_h = self.width(), self.height()
        img_w, img_h = self.image.width(), self.image.height()
        scale = min(widget_w / img_w, widget_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        self.scaled_size = (scaled_w, scaled_h)
        self.offset_x = (widget_w - scaled_w) // 2
        self.offset_y = (widget_h - scaled_h) // 2

    def resizeEvent(self, event):
        self.update_scaled_params()
        self.repaint()
        super().resizeEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image and self.scaled_size:
            painter = QPainter(self)
            scaled = self.image.scaled(*self.scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(self.offset_x, self.offset_y, scaled)
            scale_x = self.scaled_size[0] / self.image.width()
            scale_y = self.scaled_size[1] / self.image.height()
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
                    int(self.offset_x + x1 * scale_x),
                    int(self.offset_y + y1 * scale_y),
                    int((x2 - x1) * scale_x),
                    int((y2 - y1) * scale_y)
                )
                painter.drawRect(rect)
                # Draw super_row and super_column if present
                if "super_row" in shape and "super_column" in shape:
                    painter.setPen(QColor(0, 0, 0))
                    font = QFont()
                    font.setPointSize(6)
                    painter.setFont(font)
                    text = f"{shape['super_row']},\n{shape['super_column']}"
                    painter.drawText(rect, Qt.AlignCenter, text)
                if idx == self.selected_idx:
                    painter.setPen(QPen(Qt.yellow, 3, Qt.DashLine))
                    painter.drawRect(rect)
            painter.end()

    def mousePressEvent(self, event):
        if not self.image or not self.shapes or not self.scaled_size:
            return
        scale_x = self.scaled_size[0] / self.image.width()
        scale_y = self.scaled_size[1] / self.image.height()
        x = (event.x() - self.offset_x) / scale_x
        y = (event.y() - self.offset_y) / scale_y
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
        self.image_label.setMinimumWidth(600)
        self.image_label.setMaximumWidth(900)
        self.image_label.setMinimumHeight(900)
        self.image_label.boxClicked.connect(self.on_box_clicked)

        # Middle: zoomed snippet
        self.snippet_label = QLabel()
        self.snippet_label.setAlignment(Qt.AlignCenter)
        self.snippet_label.setFixedWidth(320)  # Fixed width for snippet
        self.snippet_label.setFixedHeight(640) # Fixed height for snippet
        self.snippet_label.setStyleSheet("background: #eee; border: 1px solid #aaa;")
        # Button to open another folder
        self.open_folder_btn = QPushButton("Open Another Folder")
        self.open_folder_btn.clicked.connect(self.open_another_folder)

        # Right: 3 text boxes and buttons
        self.ocr_box = LineNumberedTextEdit()
        self.ocr_box.setReadOnly(True)
        self.ocr_btn = QPushButton("Choose")
        self.ocr_btn.clicked.connect(self.choose_ocr)

        self.llm_box = LineNumberedTextEdit()
        self.llm_box.setReadOnly(True)
        self.llm_btn = QPushButton("Choose")
        self.llm_btn.clicked.connect(self.choose_llm)

        self.human_box = LineNumberedTextEdit()        
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
        layout.addWidget(self.image_label)
        # Create a vertical layout for the snippet and the button
        snippet_layout = QVBoxLayout()
        snippet_layout.addWidget(self.snippet_label)
        snippet_layout.addWidget(self.open_folder_btn)
        snippet_layout.addStretch(1)  # Push button to the top if space

        layout.addLayout(snippet_layout)
        layout.addLayout(right_layout)
        layout.setStretch(0, 3)  # Left: 3 parts
        layout.setStretch(1, 1)  # Middle: 1 part (fixed)
        layout.setStretch(2, 2)  # Right: 2 parts
        self.setLayout(layout)

        self.load_page(self.current_idx)

    def _find_jsons(self):
        jsons = []
        for root, _, files in os.walk(self.input_dir):
            for f in files:
                if f.lower().endswith(".json"):
                    jsons.append(os.path.join(root, f))
        # Sort using natural order
        jsons.sort(key=lambda x: natural_key(os.path.basename(x)))
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
            self.snippet_label.clear()
            return
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.image_label.load(img_path, self.data.get("shapes", []))
        self.ocr_box.clear()
        self.llm_box.clear()
        self.human_box.clear()
        self.snippet_label.clear()
        self.page_img_path = img_path
        self.setWindowTitle(f"OCR/LLM/Human Table Correction Tool - {os.path.basename(json_path)}")

    def on_box_clicked(self, idx):
        self.current_shape_idx = idx
        self.timer_start = time.time()
        shape = self.data["shapes"][idx]
        # Load OCR
        ocr_text = ""
        if "tesseract_output" in shape and "ocr_text" in shape["tesseract_output"]:
            ocr_text = shape["tesseract_output"]["ocr_text"]
        #if "trOCR output" in shape:
        #    ocr_text = shape["trOCR output"]

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
        # Show zoomed snippet
        self.show_snippet(shape)
        
    def open_another_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select input folder with LabelMe JSONs and images",
            os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder:
            self.input_dir = folder
            self.json_files = self._find_jsons()
            self.current_idx = 0
            self.load_page(self.current_idx)

    def show_snippet(self, shape):
        if not hasattr(self, "page_img_path"):
            self.snippet_label.clear()
            return
        if "points" not in shape or len(shape["points"]) < 2:
            self.snippet_label.clear()
            return
        try:
            pil_img = Image.open(self.page_img_path)
            img_w, img_h = pil_img.size
            x1, y1 = shape["points"][0]
            x2, y2 = shape["points"][1]
            x1, x2 = sorted([round(x1), round(x2)])
            y1, y2 = sorted([round(y1), round(y2)])
            # Clamp to image bounds
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(0, min(x2, img_w))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(0, min(y2, img_h))
            if x2 <= x1: x2 = min(x1 + 1, img_w)
            if y2 <= y1: y2 = min(y1 + 1, img_h)
            snippet = pil_img.crop((x1, y1, x2, y2))
            orig_w, orig_h = snippet.size

            # Target size for snippet label
            target_w = self.snippet_label.width()
            target_h = self.snippet_label.height()

            # Calculate zoom: up to 3x, but not exceeding label size
            zoom = min(3.0, target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * zoom)
            new_h = int(orig_h * zoom)
            snippet = snippet.resize((new_w, new_h), Image.LANCZOS)

            # Convert to QPixmap
            #snippet_qt = QPixmap.fromImage(
            #    QImage(snippet.tobytes("raw", "RGB"), snippet.width, snippet.height, QImage.Format_RGB888)
            #    if snippet.mode == "RGB"
            #    else QImage(snippet.convert("RGB").tobytes("raw", "RGB"), snippet.width, snippet.height, QImage.Format_RGB888)
            #)
            #self.snippet_label.setPixmap(snippet_qt)
            #snippet_qt = QPixmap.fromImage(ImageQt(snippet))
            #snippet_qt = pil2pixmap(snippet)
            #self.snippet_label.setPixmap(snippet_qt)
            snippet_qt = pil2pixmap(snippet)
            self.snippet_label.setPixmap(snippet_qt.scaled(
                self.snippet_label.width(),
                self.snippet_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))


        except Exception as e:
            self.snippet_label.clear()

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
    app = QApplication(sys.argv)
    input_dir = None
    if len(sys.argv) < 2:
        # Show open folder dialog if no argument is given
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select input folder with LabelMe JSONs and images",
            os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if not folder:
            print("No folder selected. Exiting.")
            sys.exit(0)
        input_dir = folder
    else:
        input_dir = sys.argv[1]
    window = MainWindow(input_dir)
    window.show()
    sys.exit(app.exec_())