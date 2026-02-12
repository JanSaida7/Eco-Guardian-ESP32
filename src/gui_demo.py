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
        
        # Visualization Frame
        self.viz_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=1)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # --- Matplotlib Setup ---
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax_wave = self.fig.add_subplot(111)
        self.ax_wave.set_title("Audio Waveform (Last 2s)")
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.grid(True)
        
        # Initial empty plot
        # Downsample x_data immediately to match update loop (factor of 10)
        self.x_data = np.linspace(0, 2, 32000)[::10] 
        self.line_wave, = self.ax_wave.plot(self.x_data, np.zeros(len(self.x_data)), linewidth=0.5)
        
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
        while not self.processor.audio_queue.empty():
            result = self.processor.process_next_chunk()
        
        # If we got a result (means new data processed)
        if result:
            probs, confidence, label, rms = result
            
            # Update Text
            if confidence > 0.6: 
                 self.result_var.set(f"Detected: {label.upper()} ({confidence:.2f})")
                 if label in ["chainsaw", "gunshot"]:
                     self.result_label.config(foreground="red")
                 else:
                     self.result_label.config(foreground="green")
            else:
                 self.result_var.set(f"Scanning... Vol: {rms:.3f}")
                 self.result_label.config(foreground="gray")

            # Update Waveform
            # Downsample for performance (plot every 10th sample)
            display_data = self.processor.audio_buffer[::10]
            
            self.line_wave.set_ydata(display_data)
            self.canvas.draw_idle() # Efficient redraw
        
        # Schedule next update
        self.root.after(50, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = EcoGuardianGUI(root)
    root.mainloop()
