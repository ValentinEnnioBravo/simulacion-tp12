from importlib.machinery import SourceFileLoader
path = r"c:\Users\nicol\Desktop\simulacion-tp12\src\simulacion\simulacionInvierno.py"
mod = SourceFileLoader("simInv", path).load_module()
Sim = mod.SimulacionInvierno
sim = Sim()
# Run 1000 days
for i in range(1000):
    sim.simular_turno()
res = sim.obtener_resultados()
print('FO days:', res.get('days_operated_on_fo'))
print('FO consumed units:', res.get('total_fo_consumed_units'))
print('FO mwh generated:', res.get('total_fo_mwh_generated'))
print('FO shortage events:', res.get('fo_shortage_events'))
print('FO reorders:', res.get('fo_reorders'))
print('PP_prom_mensual_MW =', res.get('PP_prom_mensual_MW'))
