"""
SIMULACIÓN DE PLANTA DE GENERACIÓN - INVIERNO

Implementación basada en el diagrama de flujo provisto. Esta versión reutiliza
datos de los CSVs en `public/` cuando están disponibles y aplica una lógica
de ola fría característica de invierno (aumenta la demanda en días puntuales).

Ejecutar: `python3 src/simulacion/simulacionInvierno.py`
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy import stats
import pathlib
from tabulate import tabulate

class SimulacionInvierno:
    """Simulación de planta de generación - Invierno.

    Estructura y nomenclatura alineada con `simulacionVerano.py`:
    - parámetros agrupados
    - inicialización de distribuciones en `_inicializar_distribuciones`
    - métodos de muestreo cortos: `AD()`, `DV()`, `GDCC1()`, `GDCC2()`, `GDTV()`
    """

    def __init__(self):
        # Variables de estado
        self.T = 0           # Contador de iteraciones (días)
        self.TF = 10000      # Días a simular
        self.TG = 'ESTANDAR' # Tipo de guardia: 'ESTANDAR' o 'MINIMA'
        self.RP = 0          # Contador para el reemplazo del repuesto
        self.H = 1           # Estado de la turbina de vapor (1 = habilitado, 0 = inhabilitado)
        self.GD = 0          # Producción diaria total
        self.BESS = 0        # Nivel de almacenamiento del sistema BESS
        self.BENEF = 0       # Beneficio acumulado
        self.CCC = 0         # Contador de ciclos completos
        self.CBESS = 1000    # Capacidad máxima del BESS
        self.STI = 0         # Suma de ingresos por turbina
        self.STB = 0         # Energía sobrante almacenada
        self.C = 0           # Contador de veces que se necesitó el BESS
        self.CBESS_inicial = 1000
        self.STM = 0
        self.FO = 250000          # Nivel actual de Fuel Oil
        self.MAX_FO = 500000.0  # Capacidad máxima de Fuel Oil
        self.RFO = 0               # Contador para reposición de Fuel Oil
        self.PFO = 200000.0       # Tamaño del pedido de Fuel Oil
        
        # Tracking totals for reporting
        self.total_revenue = 0.0                # Ingresos brutos por venta de energía
        self.total_revenue_from_bess = 0.0      # Ingresos obtenidos específicamente del BESS
        self.total_personal_cost = 0.0          # Costos de personal acumulados
        self.total_fuel_cost = 0.0              # Costos de combustible acumulados
        self.total_bess_amortization = 0.0      # Amortización del BESS acumulada
        self.total_fines = 0.0                  # Multas por déficit acumuladas
        
        # Parámetros económicos
        self.PV = 50         # Precio de venta por megavatio generado
        self.CC = 10         # Costo unitario del combustible
        self.CU = 5          # Costo de uso del sistema
        self.PM = 110         # Precio de multa por déficit energético
        self.I = 1           # Factor de conversión

        # --- Totales acumulados para métricas y costos ---
        self.total_costos = 0.0
        self.total_ingresos = 0.0
        self.total_generacion = 0.0
        self.total_demanda = 0.0
        self.total_mw_lost_gas_restriction = 0.0
        self.total_mw_lost_forzadas = 0.0
        self.total_cost_fallas_forzadas = 0.0
        self.total_ahorros_bess_mwh = 0.0
        self.total_ahorros_bess_value = 0.0
        self.total_multas = 0.0
        self.SAB = 0.0 # Suma Ahorro BESS (MW) - guessing naming based on context or just Initialize it to be safe
        
        # Parámetros de costos adicionales
        self.precio_cbess_por_mw = 5000.0   # Costo anual por MW de capacidad BESS (estimado)
        self.costo_adicional_por_mw_falla = 200.0 # Costo por MW perdido en falla forzada (estimado)

        # --- Parámetros climáticos/invernales ---
        self.prob_ola_frio = 0.25
        self.factor_demanda_ola_frio = 0.15  # +15% demanda en ola fría
        
        # Inicializar distribuciones tipo FDP (como en simulacionVerano)
        self._inicializar_distribuciones()

        # Resultados y series temporales
        self.resultados = []
        self.timeseries = []
    
    def _inicializar_distribuciones(self):
        """Inicializar todas las distribuciones probabilísticas"""
        
        # Autodescarga
        self.AD_dist = stats.gennorm.rvs(beta=1.6831, loc=0.00305, scale=0.00074, size=1000)
        
        # Costo Falla
        self.CF_dist = stats.pearson3.rvs(skew=-0.91, loc=0.66, scale=0.47, size=1000)
        
        # Demanda Primer Semestre
        self.DV_dist = stats.laplace_asymmetric.rvs(kappa=1.52, loc=2360.46, scale=162.14, size=1000)
        
        # Demanda Segundo Semestre
        self.DI_dist = stats.gumbel_r.rvs(loc=1734.01, scale=206.19, size=1000)
        
        # Generación diaria de CC1
        self.GD1_dist = stats.gennorm.rvs(beta=1.6831, loc=678.46, scale=164.81, size=1000)
        
        # Generación diaria de CC2
        self.GD2_dist = stats.gennorm.rvs(beta=1.6831, loc=598.27, scale=145.3, size=1000)
        
        # Generación diaria de TV
        self.GDTV_dist = stats.gennorm.rvs(beta=1.6831, loc=415.42, scale=100.9, size=1000)
        
        # Potencia Perdida
        self.PP_dist = stats.tukeylambda.rvs(lam=-0.08, loc=0.11, scale=0.03, size=1000)
        
    def AD(self):
        """Autodescarga"""
        return np.random.choice(self.AD_dist, 1)[0]
    
    def CF(self):
        """Costo Falla"""
        return np.random.choice(self.CF_dist, 1)[0]
    
    def DV(self):
        """Demanda Primer Semestre"""
        return np.random.choice(self.DV_dist, 1)[0]
    
    def DI(self):
        """Demanda Segundo Semestre"""
        return np.random.choice(self.DI_dist, 1)[0]
    
    def GDCC1(self):
        """Generación diaria de CC1"""
        return np.random.choice(self.GD1_dist, 1)[0]
    
    def GDCC2(self):
        """Generación diaria de CC2"""
        return np.random.choice(self.GD2_dist, 1)[0]
    
    def GDTV(self):
        """Generación diaria de TV"""
        return np.random.choice(self.GDTV_dist, 1)[0]
    
    def PP(self):
        """Potencia Perdida"""
        return np.random.choice(self.PP_dist, 1)[0]
    
    def CABESS(self):
        """Costo de amortización BESS"""
        return 5000

    def simular_turno(self):
        """Simula un turno (día) según el diagrama adaptado para invierno."""
        self.T += 1
        self.GD = 0.0

        # 1. Llegadas
        if self.RP == self.T:
            self.H = 1
        if self.RFO == self.T:
            self.FO = min(self.FO + self.PFO, self.MAX_FO)

        # 2. Decidir uso de gas vs fuel oil
        r = random.random()
        gas = True if r > 0.2 else False

        # 3. Generación GDCC1
        GDCC1 = self.GDCC1()
        r = random.random()
        if self.TG == 'ESTANDAR':
            if r <= 0.02:
                self.GD += GDCC1 * 0.61
            else:
                self.GD += GDCC1
        else:
            if r <= 0.03:
                self.GD += GDCC1 * 0.61
            else:
                self.GD += GDCC1
        # GDCC2 (también guardamos potencial para medir pérdidas por restricción de gas)
        GDCC2 = self.GDCC2()
        potencial_cc2 = GDCC2

        # 4. Rama principal: gas o fuel oil
        if gas:
            # operación a gas
            r = random.random()
            if self.TG == 'ESTANDAR':
                if r <= 0.03:
                    added_cc2 = GDCC2 * 0.73
                    self.GD += added_cc2
                else:
                    added_cc2 = GDCC2
                    self.GD += added_cc2
            else:
                if r <= 0.045:
                    added_cc2 = GDCC2 * 0.73
                    self.GD += added_cc2
                else:
                    added_cc2 = GDCC2
                    self.GD += added_cc2

            # En invierno no aplicamos penalización por ola de calor

            # Turbina a vapor solo si H == 1
            if self.H == 1:
                GDTV = self.GDTV()
                r = random.random()
                if self.TG == 'ESTANDAR':
                    if r <= 0.06:
                        self.GD += GDTV * 0.84
                    else:
                        self.GD += GDTV
                else:
                    if r <= 0.09:
                        self.GD += GDTV * 0.84
                    else:
                        self.GD += GDTV

            # Costos fijos operación gas
            self.BENEF -= 22000.0
            self.total_costos += 22000.0

        else:
            # operación fuel oil
            # al operar en fuel oil la capacidad efectiva de CC2 se reduce
            GDCC2 = GDCC2 * 0.7

            # Verificar stock fuel oil
            consumo_necesario = GDCC2 * 250.0
            if consumo_necesario <= self.FO:
                self.FO -= consumo_necesario
            else:
                # usar lo que hay
                if self.FO > 0:
                    GDCC2 = self.FO / 250.0
                    self.FO = 0.0
                else:
                    GDCC2 = 0.0

            r = random.random()
            if r <= 0.03:
                added_cc2 = GDCC2 * 0.73
                self.GD += added_cc2
            else:
                added_cc2 = GDCC2
                self.GD += added_cc2

            # Reposición si bajo
            if self.FO <= 2500.0:
                self.RFO = self.T + 2

            # Costos fijos operación fuel oil
            self.BENEF -= 15000.0
            self.total_costos += 15000.0

            # Registrar pérdida por restricción de gas: diferencia entre potencial y lo realmente añadido
            lost_due_gas = max(0.0, potencial_cc2 - added_cc2)
            self.total_mw_lost_gas_restriction += lost_due_gas

        # 5. Costos variables de combustible
        var_fuel_cost = (self.CC * self.GD)
        self.BENEF -= var_fuel_cost
        self.total_costos += var_fuel_cost

        # 6. Balance energía y BESS
        DV = self.DV()

        # Aplicar efecto de ola fría en invierno (aumenta la demanda)
        if random.random() < self.prob_ola_frio:
            DV = DV * (1 + self.factor_demanda_ola_frio)

        if self.GD >= DV:
            # Exceso
            ingresos = (self.PV * DV)
            self.BENEF += ingresos
            self.total_ingresos += ingresos
            exceso = self.GD - DV
            self.BESS += exceso

            if self.BESS > self.CBESS:
                # saturado
                sobrante_almacenado = self.CBESS - (self.BESS - exceso)
                self.BESS = self.CBESS
                self.STB += sobrante_almacenado
                self.CCC += 1
            else:
                # valor aproximado por almacenar/excedente (indicador)
                self.STB += (self.PV - self.CU)
                # no contabilizamos como ingreso inmediato

        else:
            # Déficit
            ingresos = (self.PV * self.GD)
            self.BENEF += ingresos
            self.total_ingresos += ingresos
            demanda_restante = DV - self.GD

            if self.BESS >= demanda_restante:
                # BESS cubre la demanda restante
                self.SAB += demanda_restante
                self.BESS -= demanda_restante
                ingreso_bess = demanda_restante * self.PV
                self.BENEF += ingreso_bess
                # registrar ahorros BESS
                self.total_ahorros_bess_mwh += demanda_restante
                self.total_ahorros_bess_value += ingreso_bess
                self.total_ingresos += ingreso_bess
            else:
                # batería insuficiente
                # usar lo que queda en la batería
                ingreso_bess = (self.BESS * self.PV)
                if self.BESS > 0:
                    self.total_ahorros_bess_mwh += self.BESS
                    self.total_ahorros_bess_value += ingreso_bess
                    self.total_ingresos += ingreso_bess
                self.BENEF += ingreso_bess
                demanda_restante -= self.BESS
                self.BESS = 0.0
                # multas por déficit
                multa = (demanda_restante * self.PM)
                self.BENEF -= multa
                self.STM += multa
                self.total_multas += multa
                self.total_costos += multa
                self.C += 1

        # 7. Mantenimiento baterías por ciclos
        if self.CCC == 100:
            self.CBESS = self.CBESS * (1 - 0.01)
            self.CCC = 0

        # 8. Cierre del día: autodescarga y amortización
        AD = self.AD()
        self.BESS = self.BESS * (1 - AD)

        # Amortización BESS
        amort = self.CABESS()
        self.BENEF -= amort
        self.total_costos += amort
        # Registrar amortización para reportes (igual que en verano)
        try:
            self.total_bess_amortization += amort
        except Exception:
            self.total_bess_amortization = amort

        # Registro STI (ingresos turbina)
        # aproximamos como ingreso proporcional al mínimo entre demanda y generación TV si H==1
        if self.H == 1:
            # calculamos aporte aproximado TV (si existe en el día)
            # para simplificar, aproximamos con una muestra
            gd_tv = self.GDTV()
            self.STI += (self.PV - self.CU) * min(DV, gd_tv) / self.I

        # 9. Falla catastrófica turbina vapor
        if random.random() <= 0.01 and self.H != 0:
            self.RP = self.T + 3
            self.H = 0
            # registrar pérdida esperada por falla (contabilizada por días siguientes también)
            # aquí contamos la pérdida inmediata estimada (si hubiese aportado TV hoy)
            potencial_tv_loss = self.GDTV()
            self.total_mw_lost_forzadas += potencial_tv_loss
            self.total_cost_fallas_forzadas += potencial_tv_loss * self.costo_adicional_por_mw_falla
            # reflejar en costos
            self.total_costos += potencial_tv_loss * self.costo_adicional_por_mw_falla

        # Guardar muestreo del día
        # Registrar serie diaria completa
        self.timeseries.append({
            'Dia': self.T,
            'Generacion_Total': float(self.GD),
            'Demanda': float(DV),
            'Perdida_Gas_Dia_MW': float(max(0.0, potencial_cc2 - (locals().get("added_cc2",0.0)))),
            'Perdida_Fallas_Dia_MW': float(0.0 if self.H==1 else 0.0),
            'BESS_Nivel': float(self.BESS),
            'Beneficio_Acumulado': float(self.BENEF),
            'Turbina_Habilitada': int(self.H),
            'FO': float(self.FO)
        })

        # Muestras resumidas (cada 10 días y primeros 10)
        if self.T % 10 == 0 or self.T <= 10:
            self.resultados.append({
                'Dia': self.T,
                'Generacion_Total': self.GD,
                'Demanda': float(DV),
                'BESS_Nivel': float(self.BESS),
                'Beneficio_Acumulado': float(self.BENEF),
                'Turbina_Habilitada': int(self.H),
                'FO': float(self.FO)
            })
        # Actualizar totales para rendimiento
        self.total_generacion += float(self.GD)
        self.total_demanda += float(DV)

    def generar_graficos(self, output_dir='results', show=True):
        """Genera y guarda gráficos útiles para la toma de decisiones.

        - Beneficio acumulado vs día
        - Nivel BESS vs día
        - Generación total vs Demanda vs día
        - Nivel Fuel Oil vs día
        - Histograma del cambio diario de beneficio
        """
        # Ajustar la interfaz para que coincida con `simulacionVerano.generar_graficos`
        resultados = None
        try:
            resultados = self.obtener_resultados()
        except Exception:
            pass

        if resultados is None:
            # Fallback a la serie interna
            if not self.timeseries:
                print('No hay datos para graficar.')
                return None
            df = pd.DataFrame(self.timeseries).sort_values('Dia')
        else:
            df = resultados.get('dataframe_resultados')
            if df is None or df.empty:
                # fallback a timeseries si dataframe_resultados está vacío
                if not self.timeseries:
                    print('No hay datos muestreados para graficar.')
                    return None
                df = pd.DataFrame(self.timeseries).sort_values('Dia')

        # Asegurar que la columna 'Dia' sea numérica
        df = df.copy()
        df['Dia'] = pd.to_numeric(df['Dia'])

        # Crear subplots 2x2 como en simulacionVerano
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        ax = axes[0, 0]
        if 'Beneficio_Acumulado' in df.columns:
            ax.plot(df['Dia'], df['Beneficio_Acumulado'], marker='o')
        ax.set_title('Beneficio Acumulado por Día')
        ax.set_xlabel('Día')
        ax.set_ylabel('Beneficio')

        ax = axes[0, 1]
        if 'BESS_Nivel' in df.columns:
            ax.plot(df['Dia'], df['BESS_Nivel'], marker='o', color='orange')
        ax.set_title('Nivel BESS por Día')
        ax.set_xlabel('Día')
        ax.set_ylabel('BESS (MW)')

        ax = axes[1, 0]
        if 'Generacion_Total' in df.columns and 'Demanda' in df.columns:
            ax.plot(df['Dia'], df['Generacion_Total'], label='Generación Total', marker='o')
            ax.plot(df['Dia'], df['Demanda'], label='Demanda', marker='x')
            ax.set_title('Generación Total vs Demanda')
            ax.set_xlabel('Día')
            ax.set_ylabel('MW')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Datos insuficientes', ha='center')

        ax = axes[1, 1]
        metrics = ['RIT', 'RCT', 'RAB', 'RCM']
        values = [resultados.get(k, 0.0) if resultados else 0.0 for k in metrics]
        ax.bar(metrics, values, color=['green', 'red', 'blue', 'purple'])
        ax.set_title('Métricas Mensuales')

        plt.tight_layout()

        try:
            plt.show()
        except Exception:
            pass

        return fig

    def CABESS(self):
        # costo amortización diario del BESS
        # - un término base (valor ejemplo)
        # - un componente proporcional a la capacidad `CBESS` y al precio por MW (`precio_cbess_por_mw`)
        # `precio_cbess_por_mw` se entiende como $ por MW por año y se amortiza diariamente.
        daily_capacity_cost = (self.CBESS * (self.precio_cbess_por_mw / 365.0))
        return 5000.0 + daily_capacity_cost

    def ejecutar(self, dias=None, tipo_guardia=None, max_fo=None, pfo=None, cbess=None, pv=None, cc=None, pm=None):
        # 1. Días
        if dias is not None:
            self.TF = int(dias)
        else:
            try:
                entrada = input('Ingrese número de días a simular [10000]: ').strip()
                if entrada:
                    self.TF = int(entrada)
            except Exception:
                pass

        # 2. Tipo de guardia
        if tipo_guardia is not None:
            self.TG = 'MINIMA' if str(tipo_guardia) == '0' else 'ESTANDAR'
        else:
            try:
                tipo = input('Tipo de guardia: ESTANDAR (1) o MINIMA (0) [1]: ').strip()
                if tipo == '0':
                    self.TG = 'MINIMA'
                else:
                    self.TG = 'ESTANDAR'
            except Exception:
                self.TG = 'ESTANDAR'

        # 3. Max Fuel Oil
        if max_fo is not None:
            self.MAX_FO = float(max_fo)
        else:
            try:
                mfo = input('Ingrese capacidad máxima de Fuel Oil [500000]: ').strip()
                if mfo:
                    self.MAX_FO = float(mfo)
            except Exception:
                self.MAX_FO = 100000.0
        # comenzar con el 50%
        self.FO = self.MAX_FO * 0.5

        # 3b. Tamaño del pedido de Fuel Oil (PFO)
        if pfo is not None:
            try:
                self.PFO = float(pfo)
            except Exception:
                pass
        else:
            try:
                pfo_in = input('Ingrese tamaño del pedido de Fuel Oil [200000.0]: ').strip()
                if pfo_in:
                    self.PFO = float(pfo_in)
            except Exception:
                # mantener valor por defecto
                pass

        # 3c. Capacidad máxima del BESS (CBESS)
        if cbess is not None:
            try:
                self.CBESS = float(cbess)
            except Exception:
                pass
        else:
            try:
                cbess_in = input('Ingrese capacidad máxima de almacenamiento BESS [200.0]: ').strip()
                if cbess_in:
                    self.CBESS = float(cbess_in)
            except Exception:
                pass

        print(f"Iniciando simulación invierno {self.TF} días (TG={self.TG})...")
        while self.T < self.TF:
            self.simular_turno()

        print('Simulación finalizada')
        return self.resumen()

    def resumen(self):
        dias = max(1, self.T)
        meses = max(1.0, dias / 30.0)

        beneficio_promedio_mensual = float(self.BENEF) / meses
        ahorro_bess_promedio_mensual = float(self.total_ahorros_bess_value) / meses
        ahorro_bess_mwh_promedio_mensual = float(self.total_ahorros_bess_mwh) / meses
        costo_fallas_promedio_mensual = float(self.total_cost_fallas_forzadas) / meses
        mw_perdidos_por_fallas_promedio_mensual = float(self.total_mw_lost_forzadas) / meses
        ingreso_promedio_mensual = float(self.total_ingresos) / meses
        multas_promedio_mensual = float(self.total_multas) / meses
        costo_promedio_mensual = float(self.total_costos) / meses
        perdida_gas_promedio_mw_mes = float(self.total_mw_lost_gas_restriction) / meses
        rendimiento_promedio_diario_mwh = float(self.total_generacion) / dias
        factor_cobertura = (float(self.total_generacion) / max(1.0, float(self.total_demanda))) if self.total_demanda > 0 else 0.0

        return {
            'beneficio_promedio_mensual_$': beneficio_promedio_mensual,
            'ahorro_bess_promedio_mensual_$': ahorro_bess_promedio_mensual,
            'ahorro_bess_mwh_promedio_mensual': ahorro_bess_mwh_promedio_mensual,
            'costo_promedio_mensual_por_fallas_$': costo_fallas_promedio_mensual,
            'mw_perdidos_por_fallas_promedio_mensual': mw_perdidos_por_fallas_promedio_mensual,
            'rendimiento_promedio_diario_mwh': rendimiento_promedio_diario_mwh,
            'factor_cobertura_total': factor_cobertura,
            'ingreso_promedio_mensual_$': ingreso_promedio_mensual,
            'multas_promedio_mensual_$': multas_promedio_mensual,
            'costo_promedio_mensual_$': costo_promedio_mensual,
            'perdida_gas_promedio_mw_mes': perdida_gas_promedio_mw_mes,
            'capacidad_final_bess': self.BESS,
            'capacidad_maxima_bess': self.CBESS,
            'energia_sobrante_total': self.STB,
            'ingresos_turbina_promedio': self.STI / dias,
            'dias_simulados': self.T,
            'resultados_muestreados': self.resultados
        }

    def obtener_resultados(self):
        """Construir un dict de resultados con las mismas claves que
        `SimulacionSistemaEnergetico.obtener_resultados` (simulación verano)
        para que el reporte final tenga el mismo formato.
        """
        dias = max(1, self.T)
        meses = max(1.0, dias / 30.0)

        total_revenue = self.total_ingresos
        total_revenue_from_bess = getattr(self, 'total_ahorros_bess_value', 0.0)
        total_bess_amortization = getattr(self, 'total_bess_amortization', 0.0)
        total_fines = getattr(self, 'total_multas', 0.0)
        total_costs = self.total_costos

        RIT = total_revenue / meses
        RCT = total_costs / meses
        ciclos = max(1, self.CCC)
        RCM = total_fines / ciclos
        RAB = (total_revenue_from_bess - total_bess_amortization) / meses

        df = pd.DataFrame(self.resultados) if self.resultados else pd.DataFrame()

        return {
            'beneficio_total': self.BENEF,
            'capacidad_final_bess': self.BESS,
            'capacidad_maxima_bess': self.CBESS,
            'energia_sobrante_total': self.STB,
            'rendimiento_promedio_diario': float(self.total_generacion) / max(1, dias),
            'dias_simulados': self.T,
            'estado_final_turbina': self.H,
            'dataframe_resultados': df,
            'RIT': RIT,
            'RCT': RCT,
            'RCM': RCM,
            'RAB': RAB,
            'total_revenue': total_revenue,
            'total_costs': total_costs,
            'total_revenue_from_bess': total_revenue_from_bess,
            'total_fines': total_fines
        }

    def generar_reporte(self):
        """Generar un reporte final tabulado con las mismas métricas
        que `simulacionVerano.generar_reporte`.
        """
        resultados = self.obtener_resultados()

        print("\n REPORTE FINAL DE LA SIMULACIÓN (INVIERNO)")
        print("=" * 50)

        metricas = [
            ["Beneficio Total", f"${resultados['beneficio_total']:,.2f}"],
            ["Capacidad Final del BESS", f"{resultados['capacidad_final_bess']:.2f} MW"],
            ["Capacidad Máxima del BESS", f"{resultados['capacidad_maxima_bess']:.2f} MW"],
            ["Energía Sobrante Total", f"{resultados['energia_sobrante_total']:.2f} MW"],
            ["Rendimiento Promedio Diario", f"${resultados['rendimiento_promedio_diario']:,.2f}"],
            ["Días Simulados", str(resultados['dias_simulados'])],
            ["Estado Final Turbina", "Habilitada" if resultados['estado_final_turbina'] == 1 else "Deshabilitada"]
        ]

        metricas.append(["RIT (Ingreso Total Prom. Mensual)", f"${resultados['RIT']:,.2f}"])
        metricas.append(["RCT (Costo Total Prom. Mensual)", f"${resultados['RCT']:,.2f}"])
        metricas.append(["RCM (Costo Multas por Ciclo)", f"${resultados['RCM']:,.2f}"])
        metricas.append(["RAB (Ahorro Prom. Mensual BESS)", f"${resultados['RAB']:,.2f}"])

        print(tabulate(metricas, headers=["Métrica", "Valor"], tablefmt="grid"))

        return resultados


if __name__ == '__main__':
    sim = SimulacionInvierno()
    resultados = sim.ejecutar()
    # Intentar generar reporte con formato igual a simulacionVerano
    try:
        sim.generar_reporte()
    except Exception:
        # fallback: imprimir el resumen simple
        print('\n--- Resumen ---')
        for k, v in resultados.items():
            if k != 'resultados_muestreados':
                print(f"{k}: {v}")
    # Generar y mostrar gráficos
    try:
        sim.generar_graficos(show=True)
    except Exception:
        pass