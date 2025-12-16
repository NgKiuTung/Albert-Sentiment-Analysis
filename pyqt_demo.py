import sys
import time
import torch
import numpy as np
from PIL import Image
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nets import ALBERT
from PyQt5.QtCore import Qt, QRect, QSize, QTimer
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap, QFont, QPalette, QBrush, QIcon
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QFileDialog, QApplication, QMessageBox, QLineEdit, QPlainTextEdit


class PyQtGui(QWidget):


    def __init__(self):
        super(PyQtGui, self).__init__()
        # self.setStyleSheet('''QWidget{background-color:url(background.jpg);}''')  # 设置界面的背景颜色
        self.resize(1420, 900)  # 界面的大小
        self.setWindowTitle("基于ALBERT的细粒度情感识别系统")  # 界面的title
        Image.open("model_data/background.jpg").convert("RGB").resize((1420, 900)).save("model_data/background.jpg")
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("model_data/background.jpg")))
        self.setPalette(palette)

        self.label_title = QLabel(self)
        self.label_title.setText("基于ALBERT的细粒度情感识别系统")
        self.label_title.setStyleSheet("color:black")
        self.label_title.setFont(QFont("Microsoft YaHei", 25, 100))
        self.label_title.move(350, 20)

        # 照片路径选择按钮
        Image.open("model_data/dir.jpg").resize((30, 30)).save("model_data/dir.jpg")
        self.btn_choose_text = QPushButton(self)
        self.btn_choose_text.setFixedSize(30, 30)
        self.btn_choose_text.move(100, 100)
        self.btn_choose_text.setIcon(QIcon("model_data/dir.jpg"))
        self.btn_choose_text.setIconSize(QSize(30, 30))
        self.btn_choose_text.clicked.connect(self.choose_text)

        self.tokenizer = BertTokenizer.from_pretrained("albert-chinese-small")
        self.text_path = None
        self.text_content = ""
        self.model = None

        self.edit_choose_text = QLineEdit(self)
        self.edit_choose_text.setFixedSize(200, 30)
        self.edit_choose_text.setText("选择文本路径")
        self.edit_choose_text.move(150, 100)

        Image.open("model_data/video.jpg").resize((30, 30)).save("model_data/video.jpg")
        self.btn_choose_model = QPushButton(self)
        self.btn_choose_model.setFixedSize(30, 30)
        self.btn_choose_model.move(550, 100)
        self.btn_choose_model.setIcon(QIcon("model_data/video.jpg"))
        self.btn_choose_model.setIconSize(QSize(30, 30))
        self.btn_choose_model.clicked.connect(self.choose_model)

        self.edit_choose_model = QLineEdit(self)
        self.edit_choose_model.setFixedSize(200, 30)
        self.edit_choose_model.setText("选择模型路径")
        self.edit_choose_model.move(600, 100)

        # 检测图片按键
        self.btn_detect_text = QPushButton(self)
        self.btn_detect_text.setText("开始检测")
        self.btn_detect_text.setStyleSheet("background-color:white")
        self.btn_detect_text.move(950, 100)
        self.btn_detect_text.clicked.connect(self.detect_text)

        self.label_time = QLabel(self)
        self.label_time.setText("用时:")
        self.label_time.setFont(QFont("Microsoft YaHei", 15, 50))
        self.label_time.move(50, 200)

        self.label_time_value = QLabel(self)
        self.label_time_value.setText("0 s")
        self.label_time_value.setFont(QFont("Microsoft YaHei", 15, 50))
        self.label_time_value.move(200, 200)

        self.label_result = QLabel(self)
        self.label_result.setText("检测结果:")
        self.label_result.setFont(QFont("Microsoft YaHei", 15, 50))
        self.label_result.move(50, 270)
        # 目标数目显示的位置
        self.label_result_value = QLabel(self)
        self.label_result_value.setText("None")
        self.label_result_value.setFont(QFont("Microsoft YaHei", 15, 50))
        self.label_result_value.move(200, 270)

        self.label_text = QLabel(self)
        self.label_text.setText("在此展示文本内容")
        self.label_text.setFont(QFont("Microsoft YaHei", 10, 100))
        self.label_text.move(400, 170)

        self.plain_text = QPlainTextEdit(self)
        self.plain_text.setFixedSize(800, 600)
        self.plain_text.setFont(QFont("Microsoft YaHei", 15, 100))
        self.plain_text.move(450, 200)

        self.device = torch.device("cuda")

    def choose_text(self):
        self.text_path, _ = QFileDialog.getOpenFileName(self, "选择邮件", "", "*.txt;;*,csv;;All Files(*)")
        if self.text_path != "":
            QMessageBox.question(self, 'Yes', '选择邮件成功!', QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)
            try:
                with open(self.text_path, "r", encoding="utf-8") as f:
                    self.text_content = f.read()
                self.plain_text.setPlainText(f"{self.text_content}")
                self.edit_choose_text.setText(f"{self.text_path}")
            except Exception as e:
                QMessageBox.question(self, 'No', f'打开邮件失败!{e}', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        else:
            QMessageBox.question(self, 'No', '选择邮件失败!', QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)

    def choose_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "*.pth;;All Files(*)")
        if model_path != "":
            self.edit_choose_model.setText(f"{model_path}")
            if "bert" in model_path:
                self.model = ALBERT(num_classes=7)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model = self.model.to(self.device)
                self.model.eval()
                QMessageBox.question(self, 'Yes', '选择模型成功!', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
            else:
                QMessageBox.question(self, 'No', '选择模型失败!', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        else:
            QMessageBox.question(self, 'No', '选择模型失败!', QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)

    def detect_text(self):
        try:
            label_names = ['like', 'sadness', 'fear', 'anger', 'disgust', 'happiness', 'surprise']
            content = self.plain_text.toPlainText()
            print(content)
            # 转英文
            start_time = time.time()
            info_dict = self.tokenizer(content, max_length=200, truncation=True, padding="max_length")
            input_ids = torch.unsqueeze(torch.LongTensor(info_dict["input_ids"]), dim=0).to(self.device)
            attention_mask = torch.unsqueeze(torch.LongTensor(info_dict["attention_mask"]), dim=0).to(self.device)
            with torch.no_grad():
                prob = self.model(input_ids, attention_mask).cpu().numpy()
                index = np.argmax(prob, axis=-1)[0]
            end_time = time.time()
            self.label_result_value.setText(f"{label_names[index]}")
            self.label_result_value.adjustSize()
            self.label_time_value.setText(f"{(end_time - start_time):.2f} s")
            self.label_time_value.adjustSize()

        except Exception as e:
            print(e.__traceback__.tb_lineno)
            print(e)
            QMessageBox.question(self, 'No', f'{e}', QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)

    def paintEvent(self, QPaintEvent):
        qp = QPainter()
        qp.begin(self)
        self.drawLines(qp)
        qp.end()

    @staticmethod
    def drawLines(qp):
        """
        画边框
        """
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(10, 10, 1400, 10)
        qp.drawLine(1400, 10, 1400, 850)
        qp.drawLine(1400, 850, 10, 850)
        qp.drawLine(10, 850, 10, 10)

    def close_demo(self):
        self.close()
        sys.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui_p = PyQtGui()
    ui_p.show()
    sys.exit(app.exec_())
