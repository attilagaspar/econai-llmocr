import sys
import os
import json
import time
import re
import base64
import io
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog, QPlainTextEdit
)
from PyQt5.QtWidgets import QTextEdit, QWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QRect, QSize
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QTextFormat
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QImage, QTextFormat
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QPoint
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PIL import Image, ImageDraw, ImageFont

"""
I am sending a table cell image consisting of vertically aligned numbers, and its OCR text. The OCR text is always the number of rows in the text. However, it is prone to errors in the actual content (i.e. mistaking one digit for the other). Please read the image yourself but always keep as many number of lines as there are in the OCR. Please only return the corrected column of numbers, no accompanying text like 'here is the corrected text' etc.
"""


try:
    import openai
except ImportError:
    openai = None
    print("Warning: openai package not installed. Install with: pip install openai")

DEFAULT_PROMPT_FROM_USER = "I am sending a table cell image consisting of vertically aligned numbers, and its OCR text. The OCR text is always correct about the structure of the text (i.e. the number of lines). However, it is prone to errors in the actual content (i.e. mistaking one digit for the other). Please read the image and use the structure from the OCR and return the best synthesis that preserves the OCR structure but has corrected content. Please only return the corrected text, no accompanying text like 'here is the corrected text' etc."
#LLM_MODEL = "gpt-4-turbo"
LLM_MODEL = "gpt-4o-mini"
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
    mouseMoved = pyqtSignal(float, float)  # x, y in image coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.shapes = []
        self.selected_idx = None
        self.scaled_size = None
        self.offset_x = 0
        self.offset_y = 0
        self.setMouseTracking(True)  # Enable mouse tracking

    def mouseMoveEvent(self, event):
        if not self.image or not self.scaled_size:
            return
        scale_x = self.scaled_size[0] / self.image.width()
        scale_y = self.scaled_size[1] / self.image.height()
        x = (event.x() - self.offset_x) / scale_x
        y = (event.y() - self.offset_y) / scale_y
        if 0 <= x < self.image.width() and 0 <= y < self.image.height():
            self.mouseMoved.emit(x, y)
        else:
            self.mouseMoved.emit(-1.0, -1.0)  # Use -1.0 instead of None
        super().mouseMoveEvent(event)



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
                
                rect = QRect(
                    int(self.offset_x + x1 * scale_x),
                    int(self.offset_y + y1 * scale_y),
                    int((x2 - x1) * scale_x),
                    int((y2 - y1) * scale_y)
                )
                
                # Determine fill color based on data availability
                fill_color = None
                has_ocr = "tesseract_output" in shape and "ocr_text" in shape["tesseract_output"]
                has_llm = "openai_output" in shape and "response" in shape["openai_output"]
                has_human = "human_output" in shape and "human_corrected_text" in shape["human_output"]
                
                if has_human:
                    fill_color = QColor(0, 200, 0, 80)  # More vivid green for human review
                elif has_llm:
                    fill_color = QColor(173, 216, 230, 80)  # Light blue for LLM data
                elif has_ocr:
                    fill_color = QColor(144, 238, 144, 80)  # Light green for OCR data
                
                # Fill the rectangle if we have a color
                if fill_color:
                    painter.setBrush(fill_color)
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(rect)
                
                # Draw the border
                label = shape.get("label", "")
                border_color = LABEL_COLORS.get(label, QColor(128, 128, 128, 120))
                pen = QPen(border_color, 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)
                
                # Draw super_row and super_column if present
                if "super_row" in shape and "super_column" in shape:
                    painter.setPen(QColor(0, 0, 0))
                    font = QFont()
                    font.setPointSize(6)
                    painter.setFont(font)
                    text = f"{shape['super_row']},\n{shape['super_column']}"
                    painter.drawText(rect, Qt.AlignCenter, text)
                    
                # Draw selection highlight
                if idx == self.selected_idx:
                    painter.setPen(QPen(Qt.yellow, 3, Qt.DashLine))
                    painter.setBrush(Qt.NoBrush)
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
        self.setMinimumSize(1400, 800)  # Prevent window deformation
        self.input_dir = input_dir
        self.json_files = self._find_jsons()
        self.current_idx = 0
        self.current_shape_idx = None
        self.timer_start = None
        self.current_snippet_image = None  # Store current snippet for API calls

        # Left: image with boxes
        self.image_label = ImageWithBoxes()
        self.image_label.setMinimumWidth(600)
        self.image_label.setMaximumWidth(900)
        self.image_label.setMinimumHeight(900)
        self.image_label.boxClicked.connect(self.on_box_clicked)

        # Middle: zoomed snippet
        self.snippet_label = QLabel()
        self.snippet_label.setAlignment(Qt.AlignRight)
        self.snippet_label.setStyleSheet("background: #eee; border: 1px solid #aaa;")
        
        # LLM response image
        self.llm_image_label = QLabel()
        self.llm_image_label.setAlignment(Qt.AlignLeft)
        self.llm_image_label.setStyleSheet("background: #f0f0f0; border: 1px solid #aaa;")
        
        # Button to open another folder
        self.open_folder_btn = QPushButton("Open Another Folder")
        self.open_folder_btn.clicked.connect(self.open_another_folder)

        # Mouse coordinate label
        self.mouse_coord_label = QLabel("")
        self.mouse_coord_label.setAlignment(Qt.AlignCenter)
        self.mouse_coord_label.setStyleSheet("color: #555; font-size: 12px;")


        # Connect mouseMoved signal
        self.image_label.mouseMoved.connect(self.update_mouse_coords)

        # Right: 3 text boxes and buttons
        self.ocr_box = LineNumberedTextEdit()
        self.ocr_box.setReadOnly(True)
        self.ocr_box.setMinimumHeight(200)
        self.ocr_box.setMaximumHeight(250)
        self.ocr_btn = QPushButton("Choose")
        self.ocr_btn.setMaximumHeight(30)  # Fixed button height
        self.ocr_btn.clicked.connect(self.choose_ocr)

        self.llm_box = LineNumberedTextEdit()
        self.llm_box.setReadOnly(True)
        self.llm_box.setMinimumHeight(300)
        self.llm_box.setMaximumHeight(350)
        self.llm_btn = QPushButton("Choose")
        self.llm_btn.setMaximumHeight(30)  # Fixed button height
        self.llm_btn.clicked.connect(self.choose_llm)

        self.human_box = LineNumberedTextEdit()
        self.human_box.setMinimumHeight(200)
        self.human_box.setMaximumHeight(250)        
        self.human_btn = QPushButton("Save")
        self.human_btn.setMaximumHeight(30)  # Fixed button height
        self.human_btn.clicked.connect(self.save_human)

        # Add prompt textbox and Send button for middle panel
        self.prompt_box = QTextEdit()
        self.prompt_box.setPlainText(DEFAULT_PROMPT_FROM_USER)
        self.prompt_box.setMaximumHeight(100)  # Limit height to keep it compact
        self.send_btn = QPushButton("Send")
        self.send_btn.setMaximumHeight(30)  # Fixed button height
        self.send_btn.clicked.connect(self.send_prompt)
        
        # Add Piecewise button
        self.piecewise_btn = QPushButton("Piecewise")
        self.piecewise_btn.setMaximumHeight(30)  # Fixed button height
        self.piecewise_btn.clicked.connect(self.send_piecewise)
        
        # Add cell height input
        self.cell_height_input = QTextEdit()
        self.cell_height_input.setPlainText("28")
        self.cell_height_input.setMaximumHeight(30)
        self.cell_height_input.setMaximumWidth(50)

        # New narrow column for super_row and super_column editing
        self.super_row_box = QTextEdit()
        self.super_row_box.setMaximumHeight(30)
        self.super_row_box.setMaximumWidth(80)
        
        self.super_column_box = QTextEdit()
        self.super_column_box.setMaximumHeight(30)
        self.super_column_box.setMaximumWidth(80)
        
        self.update_super_btn = QPushButton("Update")
        self.update_super_btn.clicked.connect(self.update_super_values)

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

        # Create narrow column layout for super_row and super_column editing
        super_layout = QVBoxLayout()
        super_layout.addWidget(QLabel("Super Row:"))
        super_layout.addWidget(self.super_row_box)
        super_layout.addWidget(QLabel("Super Column:"))
        super_layout.addWidget(self.super_column_box)
        super_layout.addWidget(self.update_super_btn)
        super_layout.addStretch(1)  # Push content to the top

        layout = QHBoxLayout()
        layout.addWidget(self.image_label)
        # Create a vertical layout for the snippet and the button
        snippet_layout = QVBoxLayout()
        
        # Create horizontal layout for the two images
        images_layout = QHBoxLayout()
        snippet_container = QVBoxLayout()
        snippet_container.addWidget(QLabel("Original"), 0)  # Label doesn't expand
        snippet_container.addWidget(self.snippet_label, 1)  # Image expands to fill space
        
        llm_container = QVBoxLayout()
        llm_container.addWidget(QLabel("LLM Response"), 0)  # Label doesn't expand
        llm_container.addWidget(self.llm_image_label, 1)  # Image expands to fill space
        
        images_layout.addLayout(snippet_container)
        images_layout.addLayout(llm_container)
        
        # Give the images layout the majority of the vertical space
        snippet_layout.addLayout(images_layout, 1)  # Stretch factor 1 for images
        snippet_layout.addWidget(self.open_folder_btn, 0)  # No stretch for button
        snippet_layout.addWidget(QLabel("Prompt:"), 0)  # No stretch for label
        snippet_layout.addWidget(self.prompt_box, 0)  # No stretch for prompt box
        
        # Create horizontal layout for Send and Piecewise buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.send_btn)
        buttons_layout.addWidget(self.piecewise_btn)
        buttons_layout.addWidget(QLabel("Cell height:"))
        buttons_layout.addWidget(self.cell_height_input)
        snippet_layout.addLayout(buttons_layout, 0)  # No stretch for buttons
        
        snippet_layout.addWidget(self.mouse_coord_label, 0)  # No stretch for label

        layout.addLayout(snippet_layout)
        layout.addLayout(super_layout)
        layout.addLayout(right_layout)
        layout.setStretch(0, 3)  # Left: 3 parts
        layout.setStretch(1, 2)  # Middle: 2 parts (100% increase)
        layout.setStretch(2, 0)  # Super column: narrow (fixed width)
        layout.setStretch(3, 1)  # Right: 1 part (reduced)
        self.setLayout(layout)
        
        # Set up keyboard shortcuts
        self.setup_shortcuts()

        self.load_page(self.current_idx)

    def setup_shortcuts(self):
        """Set up keyboard shortcuts"""
        # Ctrl+S for save
        save_shortcut = QShortcut(QKeySequence.Save, self)
        save_shortcut.activated.connect(self.save_human)

    def update_mouse_coords(self, x, y):
        if x < 0 or y < 0:  # Check for -1.0 instead of None
            self.mouse_coord_label.setText("")
        else:
            self.mouse_coord_label.setText(f"Mouse: ({int(x)}, {int(y)})")
            
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
        self.super_row_box.clear()
        self.super_column_box.clear()
        self.snippet_label.clear()
        self.llm_image_label.clear()  # Clear LLM response image
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
        
        # Create and display LLM response image if we have both LLM text and current snippet
        if llm_text and hasattr(self, 'current_snippet_image') and self.current_snippet_image:
            llm_img = self.create_llm_response_image(llm_text, self.current_snippet_image.size)
            if llm_img:
                llm_pixmap = pil2pixmap(llm_img)
                self.llm_image_label.setPixmap(llm_pixmap.scaled(
                    self.llm_image_label.width(),
                    self.llm_image_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
        else:
            self.llm_image_label.clear()
            
        # Load human
        human_text = ""
        if "human_output" in shape and "human_corrected_text" in shape["human_output"]:
            human_text = shape["human_output"]["human_corrected_text"]
        self.human_box.setPlainText(human_text)
        
        # Load super_row and super_column values
        super_row = str(shape.get("super_row", ""))
        super_column = str(shape.get("super_column", ""))
        self.super_row_box.setPlainText(super_row)
        self.super_column_box.setPlainText(super_column)
        
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

    def create_llm_response_image(self, text, target_size):
        """Create an image with the LLM response text that matches the target size"""
        if not text.strip():
            return None
            
        width, height = target_size
        
        # Create a white background image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Split text into lines
        lines = text.strip().split('\n')
        if not lines:
            return img
            
        # Try different font sizes to fit the text vertically
        font_size = max(8, min(72, height // max(1, len(lines)) - 4))
        
        # Try to load a default font, fallback to PIL default
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
        
        if font is None:
            return img
            
        # Calculate text positioning
        total_text_height = 0
        line_heights = []
        max_line_width = 0
        
        for line in lines:
            if font:
                try:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_width = bbox[2] - bbox[0]
                    line_height = bbox[3] - bbox[1]
                except:
                    # Fallback for older Pillow versions
                    line_width, line_height = draw.textsize(line, font=font)
            else:
                line_width, line_height = len(line) * 6, 12
                
            line_heights.append(line_height)
            total_text_height += line_height
            max_line_width = max(max_line_width, line_width)
        
        # Adjust font size if text doesn't fit
        if total_text_height > height - 10 and font_size > 8:
            scale_factor = (height - 10) / total_text_height
            new_font_size = max(8, int(font_size * scale_factor))
            try:
                if "arial.ttf" in str(font.path if hasattr(font, 'path') else ''):
                    font = ImageFont.truetype("arial.ttf", new_font_size)
                elif "DejaVuSans.ttf" in str(font.path if hasattr(font, 'path') else ''):
                    font = ImageFont.truetype("DejaVuSans.ttf", new_font_size)
                else:
                    font = ImageFont.load_default()
            except:
                pass
        
        # Draw the text with proper vertical spacing
        if len(lines) > 1:
            # Calculate spacing to distribute text evenly across the height
            used_height = sum(line_heights)
            available_space = height - used_height - 20  # 20px margin (10 top + 10 bottom)
            line_spacing = max(2, available_space // max(1, len(lines) - 1)) if len(lines) > 1 else 0
        else:
            line_spacing = 0
            
        y_offset = 10  # Start with top margin
        for i, line in enumerate(lines):
            if font:
                try:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_width = bbox[2] - bbox[0]
                    line_height = bbox[3] - bbox[1]
                except:
                    line_width, line_height = draw.textsize(line, font=font)
            else:
                line_width, line_height = len(line) * 6, 12
                
            x_offset = (width - line_width) // 2
            draw.text((x_offset, y_offset), line, fill='black', font=font)
            y_offset += line_height + line_spacing
            
        return img

    def show_snippet(self, shape):
        if not hasattr(self, "page_img_path"):
            self.snippet_label.clear()
            self.current_snippet_image = None
            return
        if "points" not in shape or len(shape["points"]) < 2:
            self.snippet_label.clear()
            self.current_snippet_image = None
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
            
            # Store the original snippet for API calls
            self.current_snippet_image = snippet.copy()
            
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
            self.current_snippet_image = None

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
        # Repaint the image to update colors
        self.image_label.repaint()
        self.timer_start = time.time()

    def update_super_values(self):
        """Update super_row and super_column values for the current shape"""
        if self.current_shape_idx is None:
            return
            
        shape = self.data["shapes"][self.current_shape_idx]
        
        # Get values from text boxes
        super_row_text = self.super_row_box.toPlainText().strip()
        super_column_text = self.super_column_box.toPlainText().strip()
        
        # Convert to integers if possible, otherwise store as strings
        try:
            super_row = int(super_row_text) if super_row_text else None
        except ValueError:
            super_row = super_row_text if super_row_text else None
            
        try:
            super_column = int(super_column_text) if super_column_text else None
        except ValueError:
            super_column = super_column_text if super_column_text else None
        
        # Update the shape data
        if super_row is not None:
            shape["super_row"] = super_row
        elif "super_row" in shape:
            del shape["super_row"]
            
        if super_column is not None:
            shape["super_column"] = super_column
        elif "super_column" in shape:
            del shape["super_column"]
        
        # Save JSON
        with open(self.json_files[self.current_idx], "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
            
        # Repaint the image to update the display
        self.image_label.repaint()
        
        print(f"Updated super_row: {super_row}, super_column: {super_column}")

    def send_prompt(self):
        """Send prompt, OCR text, and snippet image to OpenAI API"""
        if openai is None:
            print("Error: OpenAI package not installed")
            return
            
        if self.current_shape_idx is None:
            print("Error: No cell selected")
            return
            
        # Get API key from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            return
            
        try:
            # Set up OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Get prompt text
            prompt_text = self.prompt_box.toPlainText()
            if not prompt_text:
                prompt_text = DEFAULT_PROMPT_FROM_USER
                
            # Get OCR text
            ocr_text = self.ocr_box.toPlainText()
            
            # Get the current snippet image and encode it as base64
            image_base64 = None
            if hasattr(self, 'current_snippet_image') and self.current_snippet_image:
                # Convert PIL image to base64
                buffer = io.BytesIO()
                self.current_snippet_image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare the messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt_text} \n image:\n"  # Include OCR text in the prompt
                        }
                    ]
                }
            ]
            
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })
            messages[0]["content"].append({
                "type": "text",
                "text": f"OCR Text:\n{ocr_text}"
            })            
            print(f"Sending request to OpenAI API with model {LLM_MODEL}...")
            
            # Make API call
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=1000
            )
            
            # Extract response text
            llm_response = response.choices[0].message.content
            
            # Update the LLM output box
            self.llm_box.setPlainText(llm_response)
            
            # Create and display LLM response image
            if hasattr(self, 'current_snippet_image') and self.current_snippet_image:
                llm_img = self.create_llm_response_image(llm_response, self.current_snippet_image.size)
                if llm_img:
                    # Convert to QPixmap and display
                    llm_pixmap = pil2pixmap(llm_img)
                    self.llm_image_label.setPixmap(llm_pixmap.scaled(
                        self.llm_image_label.width(),
                        self.llm_image_label.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    ))
            
            # Save to the current shape
            shape = self.data["shapes"][self.current_shape_idx]
            if "openai_output" not in shape:
                shape["openai_output"] = {}
            shape["openai_output"]["response"] = llm_response
            shape["openai_output"]["model"] = LLM_MODEL
            shape["openai_output"]["prompt"] = prompt_text
            
            print("LLM response:", llm_response)

            # Save JSON
            with open(self.json_files[self.current_idx], "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
                
            # Repaint the image to update colors
            self.image_label.repaint()
                
            print("✓ OpenAI API response received and saved")
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # You could also show this error in a message box if preferred

    def detect_text_lines(self, image, fixed_cell_height=28):
        """
        Detect individual text lines using fixed cell height logic from add_ocr_to_layout_jsons.py
        Returns list of (top, bottom) coordinates for each line
        """
        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            opencv_image = image
            
        # Convert to grayscale
        if len(opencv_image.shape) == 3 and opencv_image.shape[2] == 3:
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = opencv_image.copy()
        
        # Create binary image (inverted so text pixels are white)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        roi_height = binary.shape[0]
        max_cells = roi_height // fixed_cell_height
        
        if max_cells == 0:
            return []
        
        # Horizontal projection
        projection = np.sum(binary, axis=1)
        
        # Score each possible center: sum of projection in the implied cell
        half = fixed_cell_height // 2
        scores = []
        for center in range(half, roi_height - half):
            top = center - half
            bottom = center + half
            score = np.sum(projection[top:bottom])
            scores.append((center, score))
        
        # Greedy selection of non-overlapping cells with best coverage
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_centers = []
        occupied = np.zeros(roi_height, dtype=bool)
        
        for center, _ in scores:
            top = max(0, center - half)
            bottom = min(roi_height, top + fixed_cell_height)
            if not occupied[top:bottom].any():
                selected_centers.append(center)
                occupied[top:bottom] = True
                if len(selected_centers) >= max_cells:
                    break
        
        # Sort centers by vertical position (top to bottom)
        selected_centers.sort()
        
        # Now for each selected center, find the actual text center of mass within that cell
        # and re-center the rectangle around it
        refined_cells = []
        
        for center in selected_centers:
            cell_top = max(0, center - half)
            cell_bottom = min(roi_height, center + half)
            
            # Get projection within this cell
            cell_projection = projection[cell_top:cell_bottom]
            
            if np.sum(cell_projection) > 0:
                # Find center of mass of text within this cell
                indices = np.arange(len(cell_projection))
                text_center_offset = np.average(indices, weights=cell_projection)
                actual_text_center = cell_top + text_center_offset
                
                # Re-center the rectangle around the actual text center
                new_top = max(0, int(actual_text_center - half))
                new_bottom = min(roi_height, new_top + fixed_cell_height)
                
                # Adjust if we hit boundaries
                if new_bottom - new_top < fixed_cell_height:
                    if new_top == 0:
                        new_bottom = min(roi_height, fixed_cell_height)
                    else:
                        new_top = max(0, roi_height - fixed_cell_height)
                
                refined_cells.append((new_top, new_bottom))
            else:
                # No text found, use original boundaries
                refined_cells.append((cell_top, cell_bottom))
        
        return refined_cells

    def update_snippet_with_progress_rectangle(self, lines, current_line_idx):
        """Update the snippet display with a red rectangle showing current progress"""
        if not hasattr(self, 'current_snippet_image') or not self.current_snippet_image:
            return
            
        # Create a copy of the current snippet for drawing
        snippet_with_progress = self.current_snippet_image.copy()
        draw = ImageDraw.Draw(snippet_with_progress)
        
        if current_line_idx < len(lines):
            top, bottom = lines[current_line_idx]
            # Draw red rectangle around current line
            draw.rectangle([0, top, snippet_with_progress.width - 1, bottom - 1], 
                         outline='red', width=2)
        
        # Convert to QPixmap and display
        snippet_qt = pil2pixmap(snippet_with_progress)
        self.snippet_label.setPixmap(snippet_qt.scaled(
            self.snippet_label.width(),
            self.snippet_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
        # Force GUI update
        QApplication.processEvents()

    def send_piecewise(self):
        """Send each line of the snippet individually to OpenAI API"""
        if openai is None:
            print("Error: OpenAI package not installed")
            return
            
        if self.current_shape_idx is None:
            print("Error: No cell selected")
            return
            
        if not hasattr(self, 'current_snippet_image') or not self.current_snippet_image:
            print("Error: No snippet image available")
            return
            
        # Get API key from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            return
            
        try:
            # Set up OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Get fixed cell height from input
            try:
                fixed_cell_height = int(self.cell_height_input.toPlainText().strip())
                if fixed_cell_height <= 0:
                    fixed_cell_height = 28
            except ValueError:
                fixed_cell_height = 28
                print(f"Invalid cell height input, using default: {fixed_cell_height}")
            
            # Detect individual lines in the snippet
            lines = self.detect_text_lines(self.current_snippet_image, fixed_cell_height)
            print(f"Detected {len(lines)} text lines with cell height {fixed_cell_height}")
            
            if not lines:
                print("No text lines detected in the image")
                return
            
            # Clear the LLM output box at start
            self.llm_box.clear()
            
            # Process each line individually
            line_responses = []
            
            for i, (top, bottom) in enumerate(lines):
                print(f"Processing line {i+1}/{len(lines)}: y={top}-{bottom}")
                
                # Show progress rectangle on current line
                self.update_snippet_with_progress_rectangle(lines, i)
                
                # Crop the line from the original snippet
                line_image = self.current_snippet_image.crop((0, top, self.current_snippet_image.width, bottom))
                
                # Convert to base64 for API
                buffer = io.BytesIO()
                line_image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Prepare message for this line
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Is this a number or a dash? Return a number or a dash, without any accompanying text like 'here is the corrected text'"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
                
                # Make API call for this line
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    max_tokens=50
                )
                
                line_response = response.choices[0].message.content.strip()
                line_responses.append(line_response)
                print(f"Line {i+1} response: {line_response}")
                
                # Update LLM output box incrementally
                combined_so_far = "\n".join(line_responses)
                self.llm_box.setPlainText(combined_so_far)
                
                # Force GUI update to show progress
                QApplication.processEvents()
            
            # Remove progress rectangle by restoring original snippet
            self.show_snippet(self.data["shapes"][self.current_shape_idx])
            
            # Final combined response
            combined_response = "\n".join(line_responses)
            
            # Update the LLM output box
            self.llm_box.setPlainText(combined_response)
            
            # Create and display LLM response image
            if hasattr(self, 'current_snippet_image') and self.current_snippet_image:
                llm_img = self.create_llm_response_image(combined_response, self.current_snippet_image.size)
                if llm_img:
                    # Convert to QPixmap and display
                    llm_pixmap = pil2pixmap(llm_img)
                    self.llm_image_label.setPixmap(llm_pixmap.scaled(
                        self.llm_image_label.width(),
                        self.llm_image_label.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    ))
            
            # Save to the current shape
            shape = self.data["shapes"][self.current_shape_idx]
            if "openai_output" not in shape:
                shape["openai_output"] = {}
            shape["openai_output"]["response"] = combined_response
            shape["openai_output"]["model"] = LLM_MODEL
            shape["openai_output"]["prompt"] = "Piecewise line-by-line processing"
            shape["openai_output"]["method"] = "piecewise"
            shape["openai_output"]["lines_detected"] = len(lines)
            
            print("Piecewise LLM response:", combined_response)

            # Save JSON
            with open(self.json_files[self.current_idx], "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
                
            # Repaint the image to update colors
            self.image_label.repaint()
                
            print("✓ Piecewise OpenAI API processing completed")
            
        except Exception as e:
            print(f"Error in piecewise processing: {e}")

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