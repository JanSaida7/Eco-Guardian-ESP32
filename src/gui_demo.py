import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from audio_processor import AudioProcessor

class EcoGuardianGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Eco-Guardian Real-Time Detection")
        self.root.geometry("1000x700") # Increased size for plots
        
        # Audio Processor
        self.processor = AudioProcessor()
        
        # UI Setup
        self.setup_ui()
        
        # Update Loop
        self.running = False
        
    def setup_ui(self):
        # Main Container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_label = ttk.Label(main_frame, text="Eco-Guardian: Forest Sound Monitor", font=("Helvetica", 16, "bold"))
        header_label.pack(pady=10)
        
        
        # Status Area (LEDs)
        self.led_frame = ttk.Frame(main_frame)
        self.led_frame.pack(pady=5)
        
        self.leds = {}
        for i, label in enumerate(["Background", "Chainsaw", "Gunshot"]):
            frame = ttk.Frame(self.led_frame)
            frame.pack(side=tk.LEFT, padx=10)
            
            canvas = tk.Canvas(frame, width=30, height=30, highlightthickness=0)
            canvas.pack()
            
            # Draw gray circle
            led = canvas.create_oval(5, 5, 25, 25, fill="gray", outline="black")
            
            lbl = ttk.Label(frame, text=label, font=("Helvetica", 8))
            lbl.pack()
            
            self.leds[label.lower()] = {"canvas": canvas, "id": led}

    def update_leds(self, detected_label):
        # Reset all to gray
        for label, data in self.leds.items():
            data["canvas"].itemconfig(data["id"], fill="gray")
        
        # Highlight detected
        if detected_label in self.leds:
            color = "green" if detected_label == "background" else "red"
            self.leds[detected_label]["canvas"].itemconfig(self.leds[detected_label]["id"], fill=color)

    def start_monitoring(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        # self.status_var.set("Status: Monitoring...") # Removed to save space
        self.processor.start_stream()
        self.log_event("Monitoring started.", "info")
        self.update_loop()

    def stop_monitoring(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        # self.status_var.set("Status: Stopped")
        self.processor.stop_stream()
        self.log_event("Monitoring stopped.", "info")
        self.update_leds(None) # Turn off LEDs

    def update_loop(self):
        if not self.running:
            return
            
        # Process audio chunk
        # Loop to drain queue so we are always up to date
        result = None
        gain = self.gain_var.get()
        while not self.processor.audio_queue.empty():
            result = self.processor.process_next_chunk(gain=gain)
        
        # If we got a result (means new data processed)
        if result:
            probs, confidence, label, rms = result
            
            # Threshold from slider
            threshold = self.threshold_var.get()

            # Update Text & LEDs
            if confidence > threshold: 
                 self.result_var.set(f"DETECTED: {label.upper()} ({confidence:.2f})")
                 self.update_leds(label)
                 
                 if label in ["chainsaw", "gunshot"]:
                     self.result_label.config(foreground="red")
                     self.log_event(f"DETECTED: {label.upper()} ({confidence:.2f})", "danger")
                 else:
                     self.result_label.config(foreground="green")
            else:
                 self.result_var.set(f"Scanning... (Vol: {rms:.3f})")
                 self.result_label.config(foreground="gray")
                 self.update_leds("background") # Default to background or none? Let's say background if low confidence means ambiguity or just silence which is background logic usually. Or maybe "None"

            # Update Waveform
            display_data = self.processor.audio_buffer[::10]
            self.line_wave.set_ydata(display_data)
            
            # Update Bars
            for bar, prob in zip(self.bars, probs):
                bar.set_height(prob)
            
            self.canvas.draw_idle() # Efficient redraw
        
        # Schedule next update
        self.root.after(50, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = EcoGuardianGUI(root)
    root.mainloop()
