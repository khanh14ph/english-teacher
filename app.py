import tkinter as tk
from PIL import Image, ImageTk
import requests, time, math, os
from threading import Thread
from scipy.io.wavfile import write
import sounddevice as sd

# url = "http://127.0.0.1:8000"
url = 'https://2b53-34-124-233-78.ngrok-free.app'
current_folder = os.path.dirname(os.path.realpath(__file__))
img_folder = os.path.join(current_folder, 'img')
audio_folder = os.path.join(current_folder, 'audio')

class FirstFrame:
    def __init__(self, app):
        self.app = app
        self.width_element = 450
        self.pos_x = (self.app.window_width - self.width_element) // 2
        self.frame = tk.Frame(self.app.root, background='#0d1b2a')
        self.frame.pack(fill='both', side='top', expand=True)
        self.entry = tk.Entry(self.frame, font=("Helvetica", 14, "normal"))
        self.frame.img = self.load_image('search.png', self.width_element, 40)
        self.entry.place(x=self.pos_x, y=50, height=50, width=self.width_element)
        self.button = tk.Button(self.frame, image=self.frame.img, command=self.search, relief='flat', borderwidth=0 , highlightthickness=0)
        self.button.place(x=self.pos_x, y=120, width=self.width_element)

    def load_image(self, path, width, height):
        img = Image.open(os.path.join(img_folder, path))
        img = img.resize((width,height))
        img = ImageTk.PhotoImage(img)
        return img
    
    def search(self):
        def run_thread():
            print("Searching")
            text = self.entry.get()
            self.button.config(state=tk.DISABLED)
            result = requests.post(url=f'{url}/phonemes', data={'text':text}).text
            result = eval(result)
            print(result['phonetics'])
            frame = SecondFrame(self.app, text, result['phonetics'])
            self.app.set_frame(frame)
        t = Thread(target=run_thread, daemon=True)
        t.start()

    def destroy(self):
        self.frame.destroy()
        

class SecondFrame:
    def __init__(self, app, text_talk, text_phoneme):
        self.app = app
        self.background_color = 'white'
        self.is_playing_record = False
        self.save_file_temp = os.path.join(audio_folder, 'test_app.wav')
        self.result_frame = None
        self.time_count = 0
        self.sample_rate = 16000
        self.max_length = 1000
        self.create_menu_bar()
        self.is_recording = False
        self.text_talk = text_talk
        self.text_phoneme = text_phoneme
        self.frame = tk.Frame(self.app.root, background=self.background_color)
        self.frame.pack(fill='both', side='top', expand=True)
        self.create_record_button()
        self.create_text()

    def create_menu_bar(self):
        self.menu_frame = tk.Frame(self.app.root, background=self.background_color, height=50)
        self.menu_frame.pack(fill='x', pady=(0, 10))
        self.menu_frame.img = self.load_image("exit.png", 20, 20)
        button = tk.Button(
            self.menu_frame, image=self.menu_frame.img, command=self.back, 
            background=self.background_color, relief='flat', borderwidth=0 , highlightthickness=0
        )
        button.pack(side='left', pady=10, padx=10)

    def back(self):
        if self.is_playing_record:
            return
        frame = FirstFrame(self.app)
        self.app.set_frame(frame)

    def create_text(self):
        # for text 
        self.text_talk_frame = tk.Text(self.frame, borderwidth=0, font=("Helvetica", 18, "normal"))
        self.text_talk_frame.insert(tk.INSERT, self.text_talk)
        num_lines = len(self.text_talk) // 30
        self.text_talk_frame.config(state=tk.DISABLED, height=num_lines)
        self.text_talk_frame.pack(side='top', anchor='nw', padx=30, pady=(40, 10))

        # for phoneme
        self.text_phoneme_frame = tk.Text(self.frame, borderwidth=0)
        self.text_phoneme_frame.insert(tk.INSERT, self.text_phoneme)
        self.text_phoneme_frame.config(state=tk.DISABLED)
        self.text_phoneme_frame.pack(side='top', anchor='nw', padx=30)

    def create_record_button(self):
        self.record_btn_size = 100
        self.record_audio = None
        self.frame.img = img = self.load_image("normal.png", self.record_btn_size,self.record_btn_size)
        self.record_btn = tk.Button(self.frame, image=img, width=self.record_btn_size, height=self.record_btn_size, relief='flat', borderwidth=0 , highlightthickness=0, command=self.handle_record_button)
        self.record_btn.config(image=img)
        self.record_btn.pack(side='bottom', pady=20)

    def load_image(self, path, width, height):
        img = Image.open(os.path.join(img_folder, path))
        img = img.resize((width,height))
        img = ImageTk.PhotoImage(img)
        return img
    
    def handle_record_button(self):
        if self.is_recording:
            self.stop_record()
        else:
            self.start_record()

    def start_record(self):
        if self.result_frame is not None:
            self.result_frame.destroy()
        if self.is_playing_record:
            return
        self.time_count = time.time()
        self.record_audio = sd.rec(self.max_length * self.sample_rate ,samplerate=self.sample_rate, channels=1)
        self.is_recording = True
        self.frame.img = self.load_image("recording.png", self.record_btn_size,self.record_btn_size)
        self.record_btn.config(image=self.frame.img)
        self.text_phoneme_frame.tag_delete("start")
        self.text_phoneme_frame.config(foreground='black')

    def stop_record(self):
        self.is_recording = False
        sd.stop()
        self.record_audio = self.record_audio[:math.ceil(time.time()-self.time_count)*self.sample_rate]
        self.time_count = 0
        write(self.save_file_temp, self.sample_rate, self.record_audio)
        self.frame.img = self.load_image("normal.png", self.record_btn_size,self.record_btn_size)
        self.record_btn.config(image=self.frame.img)
        t = Thread(target=self.submit)
        t.start()

    def submit(self):
        print("Analysing")
        text = self.text_talk
        self.record_btn.config(state=tk.DISABLED)
        with open(self.save_file_temp, 'rb') as f:
            result = requests.post(url=f'{url}/predict', data={'text':text}, files={'audio': f}).text
        result = eval(result)
        wrong_index = eval(result['wrong_index'])
        for s, e in wrong_index:
            self.text_phoneme_frame.tag_add('start', f'1.{s}', f'1.{e}')
        self.text_phoneme_frame.config(foreground='green')
        self.text_phoneme_frame.tag_config("start", foreground='red')
        self.record_btn.config(state=tk.NORMAL)
        self.create_show_result(float(result['correct_rate']))

    def play_recording(self):
        if self.is_playing_record:
            return
        print('start play recording')
        def run_thread():
            self.is_playing_record = True
            sd.playrec(self.record_audio,samplerate=self.sample_rate, channels=1)
            sd.wait()
            self.is_playing_record = False
        t = Thread(target=run_thread, daemon=True)
        t.start()
        
    def create_show_result(self, correct_rate):
        correct_rate = round(correct_rate * 100)
        font_large = ("Helvetica", 20, "normal")
        if correct_rate <= 50:
            color = '#d00000'
            temp = 'Try Again'
        elif correct_rate <= 80:
            color = '#ff9f1c'
            temp = 'Almost Correct'
        else:
            temp = 'Excellent!'
            color = '#70AD47'

        self.result_frame = tk.Frame(self.app.root, background='#CDCDCD')
        self.result_frame.place(x=0, y=self.app.window_height - 300, width=self.app.window_width, height=300)
        frame1 = tk.Frame(self.result_frame, background=self.background_color, height=75)
        frame1.pack(fill='x', pady=1)
        label = tk.Label(frame1, text=temp, foreground=color, background=self.background_color, font=font_large)
        label.pack(side='left', padx=20, pady=10)
        frame1.img = self.load_image('ear.png', width=30, height=30)
        btn = tk.Button(frame1, width=30, height=30, image=frame1.img, relief='flat', borderwidth=0 , highlightthickness=0, command=self.play_recording)
        btn.pack(side='right', padx=20)

        frame2 = tk.Frame(self.result_frame, background=self.background_color, height=225)
        frame2.pack(fill='x')
        label = tk.Label(frame2, background=self.background_color)
        label.pack(side='top', pady=10)
        label = tk.Label(frame2, text=f'You sound {correct_rate}% like a native speaker!', background=self.background_color, font=("Helvetica", 12, "normal"))
        label.place(x=20, y=10)
        frame2.img1 = self.load_image(f'try_again_btn{color}.png', 400, 75)
        btn = tk.Button(frame2, image=frame2.img1, width=400, height=75, relief='flat', borderwidth=0 , highlightthickness=0, command=self.start_record)
        btn.pack(side='top', pady=10)
        frame2.img2 = self.load_image('try_new_word_btn.png', 400, 75)
        btn = tk.Button(frame2, image=frame2.img2, width=400, height=75, relief='flat', borderwidth=0 , highlightthickness=0, command=self.back)
        btn.pack(side='top', pady=(10, 0))
        label = tk.Label(frame2)
        label.pack(pady=20)


    def destroy(self):
        self.frame.destroy()
        self.menu_frame.destroy()
        if self.result_frame is not None:
            self.result_frame.destroy()


class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("English App Using AI")
        self.window_width = 500
        self.window_height = 650
        self.root.geometry(f'{self.window_width}x{self.window_height}')
        self.root.resizable(False, False)
        self.frame = FirstFrame(self)

    def set_frame(self, frame):
        self.frame.destroy()
        self.frame = frame

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    app = Application()
    app.run()