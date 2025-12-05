"""
SIMULACIÓN DEL SISTEMA ENERGÉTICO
Basado en el diagrama de flujo especificado y las funciones de distribuciones probabilísticas

Autor: Simulación Energética
Fecha: Diciembre 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy import stats
import os
from tabulate import tabulate

GUARDIAS = ["MINIMA","ESTANDAR"]

class SimulacionSistemaEnergetico:
    def __init__(self):
        """
        Inicializar la simulación del sistema energético
        
        Args:
            dias_simulacion (int): Número de días a simular
        """
        # Variables de estado
        self.T = 0           # Contador de iteraciones (días)
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
        self.PM = 80         # Precio de multa por déficit energético
        self.I = 1           # Factor de conversión
        
        # Inicializar distribuciones probabilísticas
        self._inicializar_distribuciones()
        
        # Lista para resultados
        self.resultados_diarios = []
    
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
    
    def simular_dia(self):
        """Simular un día completo del sistema energético"""
        
        # Incrementar día
        self.T += 1
        self.GD = 0  # Reiniciar producción diaria
        
        # Verificar si llegó el repuesto
        if self.RP == self.T:
            self.H = 1  # Habilitar turbina a vapor
        
        if self.TG == "ESTANDAR":
            PFC1, PFC2, PFTV = 0.02, 0.03, 0.06
            costo_personal = 22000
        else:  # TG ≠ ESTÁNDAR
            PFC1, PFC2, PFTV = 0.03, 0.045, 0.09
            costo_personal = 15000
        
        # --- GENERACIÓN DE ENERGÍA ---
        
        # GDCC1: Producción diaria de Ciclo Combinado 1
        GDCC1_value = self.GDCC1()
        r = random.random()
        if r <= PFC1: # Falla el Ciclo Combinado 1
            self.GD += GDCC1_value * 0.61
        else:
            self.GD += GDCC1_value
        
        # GDCC2: Producción diaria de Ciclo Combinado 2
        GDCC2_value = self.GDCC2()
        r = random.random()
        if r <= PFC2: # Falla el Ciclo Combinado 2
            self.GD += GDCC2_value * 0.73
        else:
            self.GD += GDCC2_value
        
        # ¿Ola de calor?
        r = random.random()
        if r < 0.4:
            self.GD *= (1 - self.PP())
        
        # Turbina de Vapor habilitada
        GDTV_value = 0
        if self.H == 1:
            GDTV_value = self.GDTV()
            r = random.random()
            if r <= PFTV: # Falla Turbina de Vapor
                self.GD += GDTV_value * 0.84
            else:
                self.GD += GDTV_value
        
        # Aplicar costos
        self.BENEF -= costo_personal        # Costo personal 
        self.total_personal_cost += costo_personal
        self.BENEF -= self.CC * self.GD     # Costo combustible
        self.total_fuel_cost += self.CC * self.GD
        
        # --- MANEJO DEL SISTEMA DE ENERGÍA Y ALMACENAMIENTO ---
        DV_value = self.DV()
        deficit = 0
        if self.GD >= DV_value:
            self.BENEF += DV_value * self.PV
            self.total_revenue += DV_value * self.PV
            energia_sobrante = self.GD - DV_value
            self.BESS += energia_sobrante
            
            if self.BESS >= self.CBESS:
                self.BESS = self.CBESS
                self.STB += self.CBESS
                self.CCC = self.CCC + 1 # CCC: Contador Ciclo Carga

            else:
                self.STB += self.PV - self.CU 
        else:
            self.BENEF += self.PV * self.GD
            self.total_revenue += self.PV * self.GD
            deficit = DV_value - self.GD
            self.GD = 0
            
            if self.BESS >= deficit: # Alcanza
                self.BENEF += self.PV * deficit
                self.total_revenue += self.PV * deficit
                self.total_revenue_from_bess += self.PV * deficit
                self.BESS -= deficit
            else: # No, se consume lo que hay
                # Vender la energía disponible en BESS
                self.BENEF += self.PV * self.BESS
                self.total_revenue += self.PV * self.BESS
                self.total_revenue_from_bess += self.PV * self.BESS
                # Reducir el déficit por lo que se consumió del BESS
                deficit -= self.BESS
                # Aplicar multa por el déficit restante
                self.BENEF -= deficit * self.PM
                self.total_fines += deficit * self.PM
                self.BESS = 0
                self.STM = self.STM + deficit * self.PM
                self.C = self.C + 1
        
        # Contador de ciclos completos
        self.CCC += 1
        self.BENEF = self.BENEF + deficit * self.PV
        if self.CCC >= 100: # Cumplio ciclos
            self.CBESS = max(0, self.CBESS - self.CBESS_inicial * 0.01) # Reduccion por ciclos
            self.CCC = 0
        
        # Autodescarga diaria del BESS
        self.BESS *= (1 - self.AD())
        
        # Amortización del BESS
        amort = self.CABESS()
        self.BENEF -= amort
        self.total_bess_amortization += amort
        
        # Registrar ingresos de la turbina
        if self.H == 1:
            self.STI += (self.PV - self.CU) * min(DV_value, GDTV_value) / self.I
        
        # Fallo de turbina
        r = random.random()
        if r <= 0.01 and self.H == 1:
            self.RP = self.T + 3
            self.H = 0
        
        # Guardar resultados (muestreo cada 10 días)
        if self.T % 10 == 0 or self.T <= 10:
            self.resultados_diarios.append({
                'Dia': self.T,
                'Generacion_Total': self.GD,
                'Demanda': DV_value,
                'BESS_Nivel': self.BESS,
                'BESS_Capacidad': self.CBESS,
                'Beneficio_Acumulado': self.BENEF,
                'Turbina_Habilitada': self.H,
                'Tipo_Guardia': self.TG
            })
    
    def ejecutar_simulacion(self):
        """Ejecutar la simulación completa"""
        
        # Solicitar días de simulación
        while True:
            dias = input("Ingresá la cantidad de días de la simulación: ").strip()
            if dias.isdigit():
                dias = int(dias)
                break
            print("Debes ingresar un número entero.\n")
        
        # Asignar dias
        self.TF = dias

        # Solicitar tipo de simulación
        while True:
            tipo = input("Ingrese el tipo de guardia: ESTANDAR (1) o GUARDIA MINIMA (0): ").strip()
            if tipo in ("1", "0"):
                tipo = int(tipo)
                break
            print("Valor inválido. Ingrese el tipo de guardia: ESTANDAR (1) o GUARDIA MINIMA (0).\n")
        # Asignar TF
        self.TG = GUARDIAS[tipo]

        # Ingresar precio de venta disponible
        while True:
            pv = input("Ingrese el precio de venta: ").strip()
            if pv.isnumeric():
                pv = int(pv)
                break
            print("Debes ingresar un número\n")
        # Asignar TF
        self.PV = pv
        
        # Ingresar precio multa
        while True:
            pm = input("Ingrese el precio de la multa: ").strip()
            if pm.isnumeric():
                pm = int(pm)
                break
            print("Debes ingresar un número\n")
        # Asignar TF
        self.PM = pm
        
        # Ingresar capacidad del bess
        while True:
            cbess = input("Ingrese la capacidad del almacenamiento en sistema BESS: ").strip()
            if cbess.isnumeric():
                cbess = int(cbess)
                break
            print("Debes ingresar un número\n")
        # Asignar TF
        self.CBESS = cbess
        self.CBESS_inicial = cbess

        # Ejecución de la simulación
        print(f"\nIniciando simulación para {self.TF} días...\n")
        
        while self.T < self.TF:
            self.simular_dia()
        
        print("Simulación completada!\n")
        
        # Calcular indicadores finales
        self.CR = self.BENEF / self.TF

        return self.obtener_resultados()

    def ejecutar_simulacion_parametros(self, dias, tipo_guardia, precio_venta, precio_multa, capacidad_bess):
        """
        Ejecutar la simulación completa
        
        Parámetros:
        - dias (int): Cantidad de días de la simulación
        - tipo_guardia (int): Tipo de guardia - ESTANDAR (1) o GUARDIA MINIMA (0)
        - precio_venta (int): Precio de venta
        - precio_multa (int): Precio de la multa
        - capacidad_bess (int): Capacidad del almacenamiento en sistema BESS
        """
        
        # Asignar días
        self.TF = dias

        # Asignar tipo de guardia
        self.TG = GUARDIAS[tipo_guardia]

        # Asignar precio de venta
        self.PV = precio_venta
        
        # Asignar precio de multa
        self.PM = precio_multa
        
        # Asignar capacidad del BESS
        self.CBESS = capacidad_bess
        self.CBESS_inicial = capacidad_bess

        # Ejecución de la simulación
        print(f"\nIniciando simulación para {self.TF} días...\n")
        
        while self.T < self.TF:
            self.simular_dia()
        
        print("Simulación completada!\n")
        
        # Calcular indicadores finales
        self.CR = self.BENEF / self.TF

        return self.obtener_resultados()

    
    def obtener_resultados(self):
        """Obtener resultados finales de la simulación"""
        # Calcular meses (usar fracción de mes si corresponde)
        meses = max(1.0, self.TF / 30.0)

        total_revenue = self.total_revenue
        total_costs = (
            self.total_personal_cost +
            self.total_fuel_cost +
            self.total_bess_amortization +
            self.total_fines
        )

        RIT = total_revenue / meses  # Ingreso Total Promedio Mensual
        RCT = total_costs / meses    # Costo Total Promedio Mensual
        # Costo total por multas por ciclo (usar contador de ciclos completos)
        ciclos = max(1, self.CCC)
        RCM = self.total_fines / ciclos
        # Ahorro promedio mensual por BESS (ingresos desde BESS menos su amortización)
        RAB = (self.total_revenue_from_bess - self.total_bess_amortization) / meses

        return {
            'beneficio_total': self.BENEF,
            'capacidad_final_bess': self.BESS,
            'capacidad_maxima_bess': self.CBESS,
            'energia_sobrante_total': self.STB,
            'rendimiento_promedio_diario': self.CR,
            'dias_simulados': self.T,
            'estado_final_turbina': self.H,
            'dataframe_resultados': pd.DataFrame(self.resultados_diarios),
            'RIT': RIT,
            'RCT': RCT,
            'RCM': RCM,
            'RAB': RAB,
            'total_revenue': total_revenue,
            'total_costs': total_costs,
            'total_revenue_from_bess': self.total_revenue_from_bess,
            'total_fines': self.total_fines
        }
    
    def generar_reporte(self):
        """Generar reporte detallado de la simulación"""
        resultados = self.obtener_resultados()
        
        print("\n REPORTE FINAL DE LA SIMULACIÓN")
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

        # Agregar nuevas métricas solicitadas
        metricas.append(["RIT (Ingreso Total Prom. Mensual)", f"${resultados['RIT']:,.2f}"])
        metricas.append(["RCT (Costo Total Prom. Mensual)", f"${resultados['RCT']:,.2f}"])
        metricas.append(["RCM (Costo Multas por Ciclo)", f"${resultados['RCM']:,.2f}"])
        metricas.append(["RAB (Ahorro Prom. Mensual BESS)", f"${resultados['RAB']:,.2f}"])
        
        print(tabulate(metricas, headers=["Métrica", "Valor"], tablefmt="grid"))
        
        return resultados

    def generar_graficos(self, resultados=None, guardar=False, carpeta='plots'):
        """Generar gráficos de seguimiento de la simulación.

        Parámetros:
        - resultados (dict): resultado de `obtener_resultados`. Si es None, se llama internamente.
        - guardar (bool): si True, guarda los PNG en `carpeta`.
        - carpeta (str): ruta de carpeta para guardar los gráficos.
        """
        if resultados is None:
            resultados = self.obtener_resultados()

        df = resultados.get('dataframe_resultados')
        if df is None or df.empty:
            print('No hay datos muestreados para graficar. Ejecutá la simulación con suficiente muestreo.')
            return None

        # Asegurar que la columna 'Dia' sea numérica
        df = df.copy()
        df['Dia'] = pd.to_numeric(df['Dia'])

        # Preparar la carpeta para guardar
        if guardar:
            os.makedirs(carpeta, exist_ok=True)

        # Figura 1: Beneficio acumulado y BESS nivel
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        ax = axes[0, 0]
        ax.plot(df['Dia'], df['Beneficio_Acumulado'], marker='o')
        ax.set_title('Beneficio Acumulado por Día')
        ax.set_xlabel('Día')
        ax.set_ylabel('Beneficio')

        ax = axes[0, 1]
        ax.plot(df['Dia'], df['BESS_Nivel'], marker='o', color='orange')
        ax.set_title('Nivel BESS por Día')
        ax.set_xlabel('Día')
        ax.set_ylabel('BESS (MW)')

        # Figura 2: Generación total vs Demanda
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

        # Figura 3: Barras con métricas RIT, RCT, RAB, RCM
        ax = axes[1, 1]
        metrics = ['RIT', 'RCT', 'RAB', 'RCM']
        values = [resultados.get(k, 0.0) for k in metrics]
        ax.bar(metrics, values, color=['green', 'red', 'blue', 'purple'])
        ax.set_title('Métricas Mensuales')

        plt.tight_layout()

        if guardar:
            ruta = os.path.join(carpeta, 'simulacion_verano_graficos.png')
            fig.savefig(ruta)
            print(f'Gráficos guardados en: {ruta}')

        try:
            plt.show()
        except Exception:
            # En entornos sin display, show puede fallar; ignorar
            pass

        return fig

if __name__ == '__main__':
    simulacion = SimulacionSistemaEnergetico()
    resultados = simulacion.ejecutar_simulacion()
    # Generar reporte
    simulacion.generar_reporte()
    # Generar gráficos de seguimiento
    try:
        simulacion.generar_graficos(resultados, guardar=False)
    except Exception:
        pass
    print("\n Simulación completada exitosamente!")
