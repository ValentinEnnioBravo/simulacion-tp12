from importlib.machinery import SourceFileLoader
path = r"c:\Users\nicol\Desktop\simulacion-tp12\src\simulacion\simulacionInvierno.py"
mod = SourceFileLoader("simInv", path).load_module()
Sim = mod.SimulacionInvierno
sim = Sim()
for i in range(10):
    sim.simular_turno()
res = sim.obtener_resultados()
print('PP_prom_mensual_MW =', res.get('PP_prom_mensual_MW'))
sim.generar_reporte()
