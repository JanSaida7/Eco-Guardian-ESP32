import tkinter as tk
from tkinter import ttk
import threading
import time
from audio_processor import AudioProcessor # Assuming refactored logic is here

class EcoGuardianGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Eco-Guardian Real-Time Detection")
        self.root.geometry("800x600")
        
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
        
        # Placeholder for Visualizations (Will be populated in later commits)
        self.viz_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=1)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(self.viz_frame, text="Waveform Display Area (Pending Implementation)").pack(expand=True)
        
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
        result = self.processor.process_next_chunk()
        
        if result:
            probs, confidence, label, rms = result
            
            # Simple text update for now
            if confidence > 0.6: # specific threshold
                 self.result_var.set(f"Detected: {label.upper()} ({confidence:.2f})")
                 if label in ["chainsaw", "gunshot"]:
                     self.result_label.config(foreground="red")
                 else:
                     self.result_label.config(foreground="green")
            else:
                 self.result_var.set(f"Scanning... Vol: {rms:.3f}")
                 self.result_label.config(foreground="gray")
        
        # Schedule next update
        self.root.after(50, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = EcoGuardianGUI(root)
    root.mainloop()
