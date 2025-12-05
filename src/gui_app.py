
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import contextlib
import io

# Importar simulaciones
# Ajustar el path para que encuentre el paquete
sys.path.append(os.path.join(os.path.dirname(__file__)))
from simulacion import simulacionInvierno, simulacionVerano

class RedirectText(io.StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

class SimApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulación Energética - TP12")
        self.geometry("900x700")
        
        # Tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.frame_invierno = ttk.Frame(self.notebook)
        self.frame_verano = ttk.Frame(self.notebook)
        
        self.notebook.add(self.frame_invierno, text="Invierno")
        self.notebook.add(self.frame_verano, text="Verano")
        
        # Init tabs
        self.setup_invierno()
        self.setup_verano()
        
        # Area de logs común
        self.log_frame = ttk.LabelFrame(self, text="Salida / Logs")
        self.log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=15)
        self.log_text.pack(fill="both", expand=True)

    def setup_invierno(self):
        # Campos Invierno
        pnl = ttk.Frame(self.frame_invierno)
        pnl.pack(fill="x", padx=10, pady=10)
        
        # Grid layout
        row = 0
        ttk.Label(pnl, text="Días a simular:").grid(row=row, column=0, sticky="e", padx=5)
        self.i_dias = ttk.Entry(pnl)
        self.i_dias.insert(0, "365")
        self.i_dias.grid(row=row, column=1, sticky="w", padx=5)
        
        row += 1
        ttk.Label(pnl, text="Tipo de Guardia:").grid(row=row, column=0, sticky="e", padx=5)
        self.i_guardia = ttk.Combobox(pnl, values=["ESTANDAR (1)", "MINIMA (0)"])
        self.i_guardia.current(0)
        self.i_guardia.grid(row=row, column=1, sticky="w", padx=5)
        
        row += 1
        ttk.Label(pnl, text="Max Fuel Oil:").grid(row=row, column=0, sticky="e", padx=5)
        self.i_mfo = ttk.Entry(pnl)
        self.i_mfo.insert(0, "100000.0")
        self.i_mfo.grid(row=row, column=1, sticky="w", padx=5)
        
        row += 1
        ttk.Label(pnl, text="Precio Venta ($/MWh):").grid(row=row, column=0, sticky="e", padx=5)
        self.i_pv = ttk.Entry(pnl)
        self.i_pv.insert(0, "50.0")
        self.i_pv.grid(row=row, column=1, sticky="w", padx=5)
        
        row += 1
        ttk.Label(pnl, text="Costo Combustible ($/MWh):").grid(row=row, column=0, sticky="e", padx=5)
        self.i_cc = ttk.Entry(pnl)
        self.i_cc.insert(0, "10.0")
        self.i_cc.grid(row=row, column=1, sticky="w", padx=5)
        
        row += 1
        ttk.Label(pnl, text="Precio Multa ($/MWh):").grid(row=row, column=0, sticky="e", padx=5)
        self.i_pm = ttk.Entry(pnl)
        self.i_pm.insert(0, "80.0")
        self.i_pm.grid(row=row, column=1, sticky="w", padx=5)
        
        row += 2
        btn = ttk.Button(pnl, text="Ejecutar Simulación Invierno", command=self.run_invierno)
        btn.grid(row=row, column=0, columnspan=2, pady=15)
        
        btn_plots = ttk.Button(pnl, text="Ver Gráficos Generados", command=self.open_plots_folder)
        btn_plots.grid(row=row+1, column=0, columnspan=2)

    def setup_verano(self):
        # Campos Verano
        pnl = ttk.Frame(self.frame_verano)
        pnl.pack(fill="x", padx=10, pady=10)
        
        row = 0
        ttk.Label(pnl, text="Días a simular:").grid(row=row, column=0, sticky="e", padx=5)
        self.v_dias = ttk.Entry(pnl)
        self.v_dias.insert(0, "365")
        self.v_dias.grid(row=row, column=1, sticky="w", padx=5)
        
        row += 1
        ttk.Label(pnl, text="Tipo de Guardia:").grid(row=row, column=0, sticky="e", padx=5)
        self.v_guardia = ttk.Combobox(pnl, values=["ESTANDAR (1)", "MINIMA (0)"])
        self.v_guardia.current(0)
        self.v_guardia.grid(row=row, column=1, sticky="w", padx=5)
        
        row += 1
        ttk.Label(pnl, text="Capacidad BESS (MW):").grid(row=row, column=0, sticky="e", padx=5)
        self.v_cbess = ttk.Entry(pnl)
        self.v_cbess.insert(0, "1000")
        self.v_cbess.grid(row=row, column=1, sticky="w", padx=5)

        row += 1
        ttk.Label(pnl, text="Precio Venta ($/MWh):").grid(row=row, column=0, sticky="e", padx=5)
        self.v_pv = ttk.Entry(pnl)
        self.v_pv.insert(0, "50")
        self.v_pv.grid(row=row, column=1, sticky="w", padx=5)
        
        row += 1
        ttk.Label(pnl, text="Precio Multa ($/MWh):").grid(row=row, column=0, sticky="e", padx=5)
        self.v_pm = ttk.Entry(pnl)
        self.v_pm.insert(0, "80")
        self.v_pm.grid(row=row, column=1, sticky="w", padx=5)
        
        row += 2
        btn = ttk.Button(pnl, text="Ejecutar Simulación Verano", command=self.run_verano)
        btn.grid(row=row, column=0, columnspan=2, pady=15)
        
        btn_plots = ttk.Button(pnl, text="Ver Gráficos Generados", command=self.open_plots_folder)
        btn_plots.grid(row=row+1, column=0, columnspan=2)

    def run_invierno(self):
        # Obtener valores
        try:
            dias = int(self.i_dias.get())
            guardia_txt = self.i_guardia.get()
            guardia = '0' if 'MINIMA' in guardia_txt else '1'
            mfo = float(self.i_mfo.get())
            pv = float(self.i_pv.get())
            cc = float(self.i_cc.get())
            pm = float(self.i_pm.get())
        except ValueError:
            messagebox.showerror("Error", "Por favor ingresa valores numéricos válidos")
            return

        def task():
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "Iniciando Simulación Invierno...\n")
            
            sim = simulacionInvierno.SimulacionInvierno()
            
            # Redirigir stdout para la ejecucion
            with contextlib.redirect_stdout(RedirectText(self.log_text)):
                sim.ejecutar(dias=dias, tipo_guardia=guardia, max_fo=mfo, pv=pv, cc=cc, pm=pm)
            
            # Callback al main thread para graficar
            self.after(0, lambda: self.finish_invierno(sim))

        threading.Thread(target=task, daemon=True).start()

    def finish_invierno(self, sim):
        try:
            # Graficar en el main thread
            self.log_text.insert(tk.END, "\nGenerando gráficos...\n")
            sim.generar_graficos(output_dir='results', show=True)
            self.log_text.insert(tk.END, "\n=== FINALIZADO ===\n")
        except Exception as e:
            self.log_text.insert(tk.END, f"\nError al graficar: {e}\n")

    def run_verano(self):
        try:
            dias = int(self.v_dias.get())
            guardia_txt = self.v_guardia.get()
            guardia = 0 if 'MINIMA' in guardia_txt else 1
            cbess = int(self.v_cbess.get())
            pv = int(self.v_pv.get())
            pm = int(self.v_pm.get())
        except ValueError:
            messagebox.showerror("Error", "Por favor ingresa valores numéricos válidos")
            return

        def task():
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "Iniciando Simulación Verano...\n")
            
            sim = simulacionVerano.SimulacionSistemaEnergetico()
            
            # Ejecutar con parametros y captura de stdout
            with contextlib.redirect_stdout(RedirectText(self.log_text)):
                resultados = sim.ejecutar_simulacion_parametros(
                    dias=dias, 
                    tipo_guardia=guardia, 
                    precio_venta=pv, 
                    precio_multa=pm, 
                    capacidad_bess=cbess
                )
                sim.generar_reporte()

            # Callback main thread
            self.after(0, lambda: self.finish_verano(sim, resultados))

        threading.Thread(target=task, daemon=True).start()

    def finish_verano(self, sim, resultados):
        try:
            self.log_text.insert(tk.END, "\nGenerando gráficos...\n")
            # Mostrar graficos (bloqueante)
            sim.generar_graficos(resultados, guardar=True, carpeta='results_verano')
            self.log_text.insert(tk.END, "\n=== FINALIZADO ===\n")
        except Exception as e:
            self.log_text.insert(tk.END, f"\nError al graficar: {e}\n")

    def open_plots_folder(self):
        # Intentar abrir carpeta results
        path = os.path.abspath("results")
        if not os.path.exists(path):
            path = os.path.abspath("results_verano")
        
        if os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showinfo("Info", "Aun no se han generado resultados.")

if __name__ == "__main__":
    app = SimApp()
    app.mainloop()
