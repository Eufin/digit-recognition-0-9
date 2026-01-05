import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Line, Rectangle
from kivy.uix.widget import Widget
from kivy.core.window import Window
import numpy as np
import cv2
import tensorflow as tf
import tempfile
import os

# COLOR THEME
THEME = {
    'bg_dark': (0.06, 0.09, 0.16, 1),          # Slate 900 Background
    'bg_canvas': (1.0, 1.0, 1.0, 1),           # Canvas background (Putih - training compatible)
    'text_primary': (1.0, 1.0, 1.0, 1),        # Text Putih
    'accent_blue': (0.2, 0.6, 0.95, 1),        # Aksen biru
    'accent_green': (0.3, 0.85, 0.4, 1),       # Aksen hijau (konfirmasi)
    'brush_stroke': (0.0, 0.0, 0.0, 1),        # Goresan hitam (training compatible)
}

# CONSTANTS
CANVAS_DRAW_WIDTH = 15
IMAGE_SIZE = 28
MODEL_PATH = 'digit_model.h5'
TEMP_FILE_SUFFIX = '.png'

# KIVY CONFIG
Window.clearcolor = THEME['bg_dark']

class CanvasWidget(Widget):
    """Widget untuk menggambar digit dengan input sentuh"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drawing = False
        self.points = []
        self._init_canvas()
    
    def _init_canvas(self):
        """Inisialisasi kanvas dengan background"""
        with self.canvas:
            Color(*THEME['bg_canvas'])
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_rect, size=self._update_rect)
    
    def _update_rect(self, *args):
        """Update ukuran rectangle ketika widget berubah"""
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def on_touch_down(self, touch):
        """Mulai menggambar saat sentuh"""
        if self.collide_point(*touch.pos):
            self.drawing = True
            self.points = [touch.pos]
            return True
        return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        """Gambar garis saat geser"""
        if self.drawing and self.collide_point(*touch.pos):
            self.points.append(touch.pos)
            
            with self.canvas:
                Color(*THEME['brush_stroke'])
                Line(points=self.points, width=CANVAS_DRAW_WIDTH)
            return True
        return super().on_touch_move(touch)
    
    def on_touch_up(self, touch):
        """Selesai menggambar saat angkat sentuh"""
        if self.drawing:
            self.drawing = False
            return True
        return super().on_touch_up(touch)
    
    def clear_canvas(self):
        """Bersihkan kanvas dan redraw background"""
        self.canvas.clear()
        self._init_canvas()
    
    def get_image(self):
        """Export kanvas sebagai gambar grayscale untuk preprocessing"""
        with tempfile.NamedTemporaryFile(suffix=TEMP_FILE_SUFFIX, delete=False) as f:
            temp_path = f.name
        
        self.export_to_png(temp_path)
        img = cv2.imread(temp_path)
        os.unlink(temp_path)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

# DRAWING WIDGET
class DrawingWidget(BoxLayout):
    """Widget utama untuk menggambar dan mengenali digit"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.canvas.clear()
        with self.canvas.before:
            Color(*THEME['bg_dark'])
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_bg, pos=self._update_bg)
        
        self.model = None
        self._setup_ui()
        self._load_model()
    
    def _update_bg(self, *args):
        """Update background ketika ukuran berubah"""
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos
    
    def _setup_ui(self):
        """Setup semua UI components"""
        self.result_label = self._create_result_label()
        self.add_widget(self.result_label)
        
        self.canvas_widget = CanvasWidget()
        self.add_widget(self.canvas_widget)
        
        button_layout = self._create_button_layout()
        self.add_widget(button_layout)
    
    def _create_result_label(self):
        """Buat label untuk menampilkan hasil"""
        label = Label(
            text='Gambar digit (0-9)',
            font_size=28,
            size_hint_y=None,
            height=120,
            color=THEME['text_primary']
        )
        return label
    
    def _create_button_layout(self):
        """Buat layout tombol dengan style konsisten"""
        button_layout = GridLayout(cols=2, size_hint_y=None, height=100, spacing=10, padding=10)
        
        clear_btn = self._create_button('Bersihkan', self._clear_canvas_callback)
        recognize_btn = self._create_button('Kenali', self._recognize_digit_callback)
        
        button_layout.add_widget(clear_btn)
        button_layout.add_widget(recognize_btn)
        
        return button_layout
    
    def _create_button(self, text, callback):
        """Buat button dengan style konsisten"""
        btn = Button(text=text, size_hint_x=0.5)
        btn.background_color = THEME['accent_blue']
        btn.color = THEME['text_primary']
        btn.font_size = '18sp'
        btn.bind(on_press=callback)
        return btn
    
    def _load_model(self):
        """Load model pengenalan digit"""
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("✓ Model berhasil dimuat.")
        except FileNotFoundError:
            print("✗ Model tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
            self.result_label.text = 'Model tidak tersedia'
    
    def _clear_canvas_callback(self, instance):
        """Callback untuk tombol bersihkan"""
        self.canvas_widget.clear_canvas()
        self.result_label.text = 'Gambar digit (0-9)'
    
    def _recognize_digit_callback(self, instance):
        """Callback untuk tombol kenali"""
        if self.model is None:
            self.result_label.text = 'Model tidak tersedia'
            return
        
        img = self.canvas_widget.get_image()
        if img is None:
            self.result_label.text = 'Gambar tidak valid'
            return
        
        processed_img = self._preprocess_image(img)
        predictions = self.model.predict(processed_img, verbose=0)
        prediction = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        self.result_label.text = f'Digit: {prediction}  |  Akurasi: {confidence:.1f}%'
    
    def _preprocess_image(self, img):
        """Preprocess gambar untuk CNN model"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        resized = self._extract_digit(thresh, contours)
        
        normalized = resized.astype('float32') / 255.0
        input_img = normalized.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
        
        return input_img
    
    def _extract_digit(self, thresh, contours):
        """Ekstrak dan process digit dari kontour"""
        if not contours:
            return cv2.resize(thresh, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w > 5 and h > 5 and cv2.contourArea(cnt) > 30:
            digit = thresh[
                max(0, y-2):min(thresh.shape[0], y+h+2),
                max(0, x-2):min(thresh.shape[1], x+w+2)
            ]
            
            h_d, w_d = digit.shape
            side = max(w_d, h_d)
            padded = np.zeros((side + 10, side + 10), dtype=np.uint8)
            offset_x = (side - w_d + 10) // 2
            offset_y = (side - h_d + 10) // 2
            padded[offset_y:offset_y+h_d, offset_x:offset_x+w_d] = digit
            
            resized = cv2.resize(padded, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        else:
            resized = cv2.resize(thresh, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        
        return resized


# APP
class DigitRecognitionApp(App):
    """Main application untuk digit recognition"""
    
    def build(self):
        self.title = 'Digit Recognition App'
        return DrawingWidget()


if __name__ == '__main__':
    DigitRecognitionApp().run()
