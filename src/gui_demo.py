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
        
        # Status Area
        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("Helvetica", 10))
        status_label.pack(pady=5)
        
        # Controls Frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(pady=10)
        
        self.start_button = ttk.Button(controls_frame, text="Start Monitoring", command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(controls_frame, text="Stop Monitoring", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(controls_frame, text="Threshold:").pack(side=tk.LEFT, padx=(20, 5))
        self.threshold_var = tk.DoubleVar(value=0.6)
        self.threshold_slider = ttk.Scale(controls_frame, from_=0.0, to=1.0, variable=self.threshold_var, orient=tk.HORIZONTAL)
        self.threshold_slider.pack(side=tk.LEFT, padx=5)
        
        self.threshold_label = ttk.Label(controls_frame, textvariable=self.threshold_var)
        self.threshold_var.trace("w", lambda *args: self.threshold_label.config(text=f"{self.threshold_var.get():.2f}"))
        self.threshold_label.pack(side=tk.LEFT)

        # Gain Control
        ttk.Label(controls_frame, text="Gain:").pack(side=tk.LEFT, padx=(20, 5))
        self.gain_var = tk.DoubleVar(value=1.0)
        self.gain_slider = ttk.Scale(controls_frame, from_=0.1, to=5.0, variable=self.gain_var, orient=tk.HORIZONTAL)
        self.gain_slider.pack(side=tk.LEFT, padx=5)
        
        self.gain_label = ttk.Label(controls_frame, textvariable=self.gain_var)
        self.gain_var.trace("w", lambda *args: self.gain_label.config(text=f"x{self.gain_var.get():.1f}"))
        self.gain_label.pack(side=tk.LEFT)

        # Visualization Frame
        self.viz_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=1)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # --- Matplotlib Setup ---
        self.fig = Figure(figsize=(8, 6), dpi=100)
        
        # Waveform Plot (Top)
        self.ax_wave = self.fig.add_subplot(211)
        self.ax_wave.set_title("Audio Waveform (Last 2s) - Live")
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.grid(True, linestyle='--', alpha=0.5)
        
        # Initial empty wave
        self.x_data = np.linspace(0, 2, 32000)[::10] 
        self.line_wave, = self.ax_wave.plot(self.x_data, np.zeros(len(self.x_data)), linewidth=0.5, color='blue')
        
        # Probability Bar Chart (Bottom)
        self.ax_bar = self.fig.add_subplot(212)
        self.ax_bar.set_title("Class Probabilities")
        self.ax_bar.set_ylim(0, 1)
        self.bar_labels = ["Background", "Chainsaw", "Gunshot"]
        self.bar_colors = ["green", "red", "orange"] # Background=Green, Danger=Red/Orange
        self.bars = self.ax_bar.bar(self.bar_labels, [0, 0, 0], color=self.bar_colors)
        
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Detection Result Area
        self.result_var = tk.StringVar(value="Waiting for audio...")
        self.result_label = ttk.Label(main_frame, textvariable=self.result_var, font=("Helvetica", 14, "bold"), foreground="gray")
        self.result_label.pack(pady=20)

    def start_monitoring(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Status: Monitoring...")
        self.processor.start_stream()
        self.update_loop()

    def stop_monitoring(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Status: Stopped")
        self.processor.stop_stream()

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

            # Update Text
            if confidence > threshold: 
                 self.result_var.set(f"DETECTED: {label.upper()} ({confidence:.2f})")
                 if label in ["chainsaw", "gunshot"]:
                     self.result_label.config(foreground="red")
                     # Beep for danger
                     if confidence > 0.8:
                         # Non-blocking beep not easily possible without threading or windows specific
                         pass
                 else:
                     self.result_label.config(foreground="green")
            else:
                 self.result_var.set(f"Scanning... (Vol: {rms:.3f})")
                 self.result_label.config(foreground="gray")

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
