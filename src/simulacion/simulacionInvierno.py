"""
Simulaci贸n de Planta H铆brida de Generaci贸n Energ茅tica con Sistema BESS
======================================================================

Autor: Simulaci贸n basada en diagrama de flujo
Descripci贸n: Modelo de simulaci贸n estoc谩stica de una planta h铆brida de generaci贸n
            que incluye ciclos combinados, turbina de vapor y sistema de almacenamiento
            de bater铆as (BESS) con gesti贸n de combustibles y eventos de falla.

Distribuciones de Probabilidad Implementadas:
- Autodescarga: Generalized Normal Distribution
- Costo Falla: Pearson Type III Distribution  
- Demanda Verano: Laplace Asymmetric Distribution
- Demanda Invierno: Gumbel Right Distribution
- Generaci贸n CC1/CC2/TV: Generalized Normal Distribution
- Potencia Perdida: Tukey Lambda Distribution
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

class PlantaHibrida:
    """
    Simulaci贸n de planta h铆brida de generaci贸n energ茅tica con sistema de almacenamiento de bater铆as (BESS)
    
    La simulaci贸n modela:
    - Generaci贸n a partir de 2 ciclos combinados y 1 turbina de vapor
    - Sistema de almacenamiento BESS con carga/descarga autom谩tica
    - Gesti贸n de combustibles (gas natural y fuel oil)
    - Eventos de falla estoc谩sticos
    - Balance econ贸mico diario con multas por d茅ficit
    - Demanda estacional variable
    """
    
    def __init__(self, tiempo_final=365):
        """
        Inicializa la simulaci贸n con todas las distribuciones de probabilidad y variables
        
        Args:
            tiempo_final (int): N煤mero de d铆as a simular (por defecto 365)
        """
        self.TF = tiempo_final  # Tiempo final de simulaci贸n (d铆as)
        
        print("馃攱 Inicializando distribuciones de probabilidad...")
        
        # Generaci贸n de distribuciones de probabilidad (1000 valores pre-calculados)
        # Autodescarga de bater铆a (porcentaje diario)
        self.AD_dist = stats.gennorm.rvs(beta=1.6831, loc=0.00305, scale=0.00074, size=1000)
        
        # Costo de falla (factor multiplicador)
        self.CF_dist = stats.pearson3.rvs(skew=-0.91, loc=0.66, scale=0.47, size=1000)
        
        # Demanda Primer Semestre - Verano (MWh)
        self.DV_dist = stats.laplace_asymmetric.rvs(kappa=1.52, loc=2360.46, scale=162.14, size=1000)
        
        # Demanda Segundo Semestre - Invierno (MWh)
        self.DI_dist = stats.gumbel_r.rvs(loc=1734.01, scale=206.19, size=1000)
        
        # Generaci贸n diaria de Ciclo Combinado 1 (MWh)
        self.GD1_dist = stats.gennorm.rvs(beta=1.6831, loc=678.46, scale=164.81, size=1000)
        
        # Generaci贸n diaria de Ciclo Combinado 2 (MWh)
        self.GD2_dist = stats.gennorm.rvs(beta=1.6831, loc=598.27, scale=145.3, size=1000)
        
        # Generaci贸n diaria de Turbina de Vapor (MWh)
        self.GDTV_dist = stats.gennorm.rvs(beta=1.6831, loc=415.42, scale=100.9, size=1000)
        
        # Potencia Perdida por fallas (porcentaje 0-1)
        self.PP_dist = stats.tukeylambda.rvs(lam=-0.08, loc=0.11, scale=0.03, size=1000)
        
        # Inicializaci贸n de variables de estado
        self.inicializar_variables()
        
    def inicializar_variables(self):
        """Inicializa todas las variables de estado de la simulaci贸n"""
        print("鈿欙笍 Inicializando variables de estado...")
        
        # Variable temporal
        self.T = 0
        
        # Recursos y combustibles
        self.FO = 5000  # Fuel Oil inicial (litros)
        self.MAX_FO = 10000  # Capacidad m谩xima de Fuel Oil (litros)
        
        # Sistema de almacenamiento BESS
        self.BESS = 500  # Nivel inicial de bater铆a (MWh)
        self.CBESS = 1000  # Capacidad m谩xima de bater铆a (MWh)
        self.CCC = 0  # Contador de ciclos de carga
        
        # Estados operacionales de equipos
        self.TV_habilitada = True  # Turbina de vapor habilitada
        self.estado_CC1 = True  # Ciclo combinado 1 operativo
        self.estado_CC2 = True  # Ciclo combinado 2 operativo
        
        # Variables econ贸micas
        self.BENEF = 0  # Beneficio acumulado ($)
        self.PV = 120  # Precio de venta de energ铆a ($/MWh)
        self.PM = 150  # Precio de multa por d茅ficit ($/MWh)
        self.costo_fijo_diario = 50000  # Costos fijos diarios ($)
        self.costo_variable_combustible = 25  # Costo variable por MWh generado ($/MWh)
        
        # Contadores de llegada de recursos
        self.RP = 0  # D铆as restantes para llegada de repuestos
        self.RFO = 0  # D铆as restantes para llegada de Fuel Oil
        
        # Variables de resultado acumuladas
        self.generacion_total = 0
        self.energia_vendida = 0
        self.multas_total = 0
        self.eventos_falla = 0
        self.energia_almacenada = 0
        self.energia_desperdiciada = 0
        
        # Registro diario para an谩lisis posterior
        self.registro_diario = []
    
    # ==========================================================================
    # M脡TODOS PARA OBTENER VALORES DE DISTRIBUCIONES ESTOC脕STICAS
    # ==========================================================================
    
    def AD(self):
        """Autodescarga de bater铆a (porcentaje diario)"""
        return abs(np.random.choice(self.AD_dist, 1)[0])
    
    def CF(self):
        """Costo de falla (factor multiplicador)"""
        return abs(np.random.choice(self.CF_dist, 1)[0])
    
    def DV(self):
        """Demanda Primer Semestre - Verano (MWh)"""
        return abs(np.random.choice(self.DV_dist, 1)[0])
    
    def DI(self):
        """Demanda Segundo Semestre - Invierno (MWh)"""
        return abs(np.random.choice(self.DI_dist, 1)[0])
    
    def GDCC1(self):
        """Generaci贸n diaria de Ciclo Combinado 1 (MWh)"""
        return abs(np.random.choice(self.GD1_dist, 1)[0])
    
    def GDCC2(self):
        """Generaci贸n diaria de Ciclo Combinado 2 (MWh)"""
        return abs(np.random.choice(self.GD2_dist, 1)[0])
    
    def GDTV(self):
        """Generaci贸n diaria de Turbina de Vapor (MWh)"""
        return abs(np.random.choice(self.GDTV_dist, 1)[0])
    
    def PP(self):
        """Potencia Perdida por fallas (porcentaje 0-1)"""
        return max(0, min(1, abs(np.random.choice(self.PP_dist, 1)[0])))
    
    def CABESS(self):
        """Costo de amortizaci贸n diario del sistema BESS ($)"""
        return 5000
    
    # ==========================================================================
    # M脡TODOS DE L脫GICA DE SIMULACI脫N
    # ==========================================================================
    
    def determinar_tipo_generacion(self):
        """
        Determina si la generaci贸n usa gas natural (EST脕NDAR) o fuel oil (NO EST脕NDAR)
        
        Returns:
            str: "ESTANDAR" (70% probabilidad) o "NO_ESTANDAR" (30% probabilidad)
        """
        return "ESTANDAR" if np.random.random() < 0.7 else "NO_ESTANDAR"
    
    def determinar_demanda_diaria(self):
        """
        Calcula la demanda diaria seg煤n la temporada
        
        Returns:
            float: Demanda en MWh (mayor en verano, menor en invierno)
        """
        # Primer semestre (d铆as 1-182): verano - mayor demanda
        # Segundo semestre (d铆as 183-365): invierno - menor demanda
        if self.T <= 182:
            return self.DV()  # Demanda de verano
        else:
            return self.DI()  # Demanda de invierno
    
    def calcular_generacion_diaria(self):
        """
        Calcula la generaci贸n total del d铆a considerando:
        - Tipo de combustible (gas natural vs fuel oil)
        - Fallas en equipos con probabilidades espec铆ficas
        - P茅rdidas de potencia en caso de falla
        
        Returns:
            float: Generaci贸n total del d铆a en MWh
        """
        tipo_gen = self.determinar_tipo_generacion()
        generacion_total = 0
        
        # Factor de penalizaci贸n por tipo de combustible
        factor_combustible = 0.95 if tipo_gen == "NO_ESTANDAR" else 1.0
        
        # Generaci贸n Ciclo Combinado 1 (si est谩 operativo)
        if self.estado_CC1:
            gen_cc1 = self.GDCC1() * factor_combustible
            # Probabilidad de falla 5%
            if np.random.random() > 0.05:
                generacion_total += gen_cc1
            else:
                self.eventos_falla += 1
                # En caso de falla, se pierde un porcentaje de la generaci贸n
                generacion_total += gen_cc1 * (1 - self.PP())
        
        # Generaci贸n Ciclo Combinado 2 (si est谩 operativo)
        if self.estado_CC2:
            gen_cc2 = self.GDCC2() * factor_combustible
            # Probabilidad de falla 5%
            if np.random.random() > 0.05:
                generacion_total += gen_cc2
            else:
                self.eventos_falla += 1
                generacion_total += gen_cc2 * (1 - self.PP())
        
        # Generaci贸n Turbina de Vapor (si est谩 habilitada)
        if self.TV_habilitada:
            gen_tv = self.GDTV() * factor_combustible
            # Probabilidad de falla 3% (menor que ciclos combinados)
            if np.random.random() > 0.03:
                generacion_total += gen_tv
            else:
                self.eventos_falla += 1
                generacion_total += gen_tv * (1 - self.PP())
        
        # Consumo de combustible si se usa fuel oil
        if tipo_gen == "NO_ESTANDAR":
            combustible_usado = generacion_total * 0.3  # Factor de conversi贸n fuel oil
            self.FO = max(0, self.FO - combustible_usado)
        
        return generacion_total
    
    def gestionar_bess(self, exceso_energia, deficit_energia):
        """
        Gestiona el sistema de almacenamiento de energ铆a BESS
        
        Args:
            exceso_energia (float): Energ铆a excedente para almacenar (MWh)
            deficit_energia (float): Energ铆a faltante para cubrir (MWh)
            
        Returns:
            tuple: (energia_suministrada_por_bess, energia_almacenada_en_bess)
        """
        energia_de_bess = 0
        energia_a_bess = 0
        
        # Autodescarga diaria de la bater铆a
        autodescarga = self.BESS * self.AD()
        self.BESS = max(0, self.BESS - autodescarga)
        
        if exceso_energia > 0:
            # Hay exceso de energ铆a - cargar bater铆a si hay espacio disponible
            espacio_disponible = self.CBESS - self.BESS
            energia_a_almacenar = min(exceso_energia, espacio_disponible)
            energia_a_bess = energia_a_almacenar
            self.BESS += energia_a_almacenar
            self.energia_almacenada += energia_a_almacenar
            
            # Energ铆a que no se puede almacenar se desperdicia
            self.energia_desperdiciada += (exceso_energia - energia_a_almacenar)
            
            # Contabilizar ciclo de carga
            if energia_a_almacenar > 0:
                self.CCC += 1
        
        elif deficit_energia > 0:
            # Hay d茅ficit de energ铆a - usar bater铆a si est谩 disponible
            energia_disponible_bess = self.BESS
            energia_de_bess = min(deficit_energia, energia_disponible_bess)
            self.BESS = max(0, self.BESS - energia_de_bess)
            
            # Contabilizar ciclo de descarga
            if energia_de_bess > 0:
                self.CCC += 1
        
        # Degradaci贸n de la bater铆a cada 100 ciclos
        if self.CCC >= 100:
            degradacion = self.CBESS * 0.02  # 2% de degradaci贸n por cada 100 ciclos
            self.CBESS = max(self.CBESS - degradacion, self.CBESS * 0.6)  # M谩ximo 40% degradaci贸n total
            self.CCC = 0  # Reiniciar contador
        
        return energia_de_bess, energia_a_bess
    
    def calcular_beneficios(self, energia_generada, energia_vendida, deficit_final):
        """
        Calcula los beneficios econ贸micos del d铆a
        
        Args:
            energia_generada (float): Energ铆a total generada (MWh)
            energia_vendida (float): Energ铆a efectivamente vendida (MWh)
            deficit_final (float): D茅ficit no cubierto (MWh)
            
        Returns:
            float: Beneficio neto del d铆a ($)
        """
        # Ingresos por venta de energ铆a
        ingresos = energia_vendida * self.PV
        
        # Multas por d茅ficit energ茅tico
        multas = deficit_final * self.PM
        self.multas_total += multas
        
        # Costos operacionales
        costos_fijos = self.costo_fijo_diario
        costos_variables = energia_generada * self.costo_variable_combustible
        costo_amortizacion = self.CABESS()
        
        # Beneficio neto del d铆a
        beneficio_diario = ingresos - multas - costos_fijos - costos_variables - costo_amortizacion
        self.BENEF += beneficio_diario
        
        return beneficio_diario
    
    def gestionar_eventos_especiales(self):
        """
        Gestiona eventos especiales como fallas catastr贸ficas y llegadas de recursos
        """
        # Evento catastr贸fico en turbina de vapor (probabilidad 0.1% diaria)
        if self.TV_habilitada and np.random.random() < 0.001:
            self.TV_habilitada = False
            self.RP = 30  # 30 d铆as para reparaci贸n
        
        # Verificar llegada de repuestos
        if self.RP > 0:
            self.RP -= 1
            if self.RP == 0:
                self.TV_habilitada = True
        
        # Solicitud de fuel oil si el nivel est谩 bajo
        if self.FO < self.MAX_FO * 0.2 and self.RFO == 0:  # Menos del 20%
            self.RFO = np.random.randint(3, 8)  # Entre 3 y 7 d铆as para entrega
        
        # Verificar llegada de fuel oil
        if self.RFO > 0:
            self.RFO -= 1
            if self.RFO == 0:
                self.FO = self.MAX_FO  # Llenar tanque completamente
    
    def simular_dia(self):
        """
        Simula un d铆a completo de operaci贸n de la planta
        
        Ejecuta el siguiente flujo:
        1. Avanzar tiempo
        2. Determinar demanda del d铆a
        3. Calcular generaci贸n total
        4. Realizar balance energ茅tico
        5. Gestionar sistema BESS
        6. Calcular beneficios econ贸micos
        7. Gestionar eventos especiales
        8. Registrar datos del d铆a
        """
        # Avanzar tiempo
        self.T += 1
        
        # Determinar demanda del d铆a seg煤n temporada
        demanda = self.determinar_demanda_diaria()
        
        # Calcular generaci贸n total considerando fallas
        generacion = self.calcular_generacion_diaria()
        self.generacion_total += generacion
        
        # Balance energ茅tico: generaci贸n - demanda
        balance = generacion - demanda
        
        if balance >= 0:
            # Hay exceso de energ铆a o balance exacto
            exceso = balance
            deficit = 0
            energia_de_bess, energia_a_bess = self.gestionar_bess(exceso, 0)
            energia_vendida = demanda  # Se vende exactamente lo que se demanda
            deficit_final = 0
        else:
            # Hay d茅ficit de energ铆a
            deficit = -balance
            exceso = 0
            energia_de_bess, energia_a_bess = self.gestionar_bess(0, deficit)
            energia_vendida = generacion + energia_de_bess
            deficit_final = deficit - energia_de_bess  # D茅ficit no cubierto por BESS
        
        self.energia_vendida += energia_vendida
        
        # Calcular beneficios del d铆a
        beneficio = self.calcular_beneficios(generacion, energia_vendida, deficit_final)
        
        # Gestionar eventos especiales
        self.gestionar_eventos_especiales()
        
        # Registrar datos del d铆a para an谩lisis posterior
        self.registro_diario.append({
            'dia': self.T,
            'demanda': demanda,
            'generacion': generacion,
            'energia_vendida': energia_vendida,
            'deficit': deficit_final,
            'nivel_bess': self.BESS,
            'capacidad_bess': self.CBESS,
            'fuel_oil': self.FO,
            'beneficio': beneficio,
            'tv_operativa': self.TV_habilitada,
            'balance': balance,
            'energia_de_bess': energia_de_bess,
            'energia_a_bess': energia_a_bess
        })
    
    def ejecutar_simulacion(self):
        """
        Ejecuta la simulaci贸n completa por el n煤mero de d铆as especificado
        """
        print(f"鈿� Ejecutando simulaci贸n por {self.TF} d铆as...")
        
        for dia in range(self.TF):
            self.simular_dia()
            
            # Mostrar progreso cada 50 d铆as
            if (dia + 1) % 50 == 0:
                print(f"馃搳 Progreso: D铆a {dia + 1}/{self.TF} - Beneficio acumulado: ${self.BENEF:,.2f}")
        
        print("鉁� Simulaci贸n completada!")
    
    def obtener_resultados(self):
        """
        Retorna un resumen completo de los resultados de la simulaci贸n
        
        Returns:
            tuple: (diccionario_resultados, dataframe_registro_diario)
        """
        df_registro = pd.DataFrame(self.registro_diario)
        
        # Calcular m茅tricas de resumen
        resultados = {
            'dias_simulados': self.TF,
            'generacion_total_mwh': self.generacion_total,
            'energia_vendida_mwh': self.energia_vendida,
            'beneficio_total': self.BENEF,
            'beneficio_promedio_diario': self.BENEF / self.TF,
            'multas_total': self.multas_total,
            'eventos_falla': self.eventos_falla,
            'energia_almacenada_mwh': self.energia_almacenada,
            'energia_desperdiciada_mwh': self.energia_desperdiciada,
            'capacidad_final_bess': self.CBESS,
            'nivel_final_bess': self.BESS,
            'fuel_oil_restante': self.FO,
            'disponibilidad_promedio': (self.TF - sum(1 for r in self.registro_diario if not r['tv_operativa'])) / self.TF * 100,
            'deficit_promedio': df_registro['deficit'].mean(),
            'exceso_promedio': df_registro[df_registro['balance'] > 0]['balance'].mean() if len(df_registro[df_registro['balance'] > 0]) > 0 else 0
        }
        
        return resultados, df_registro
    
    def generar_reporte(self, guardar_archivo=True):
        """
        Genera un reporte completo de la simulaci贸n con gr谩ficos y estad铆sticas
        
        Args:
            guardar_archivo (bool): Si True, guarda los gr谩ficos como archivos PNG
        """
        resultados, registro_df = self.obtener_resultados()
        
        print("\n" + "="*80)
        print("馃搳 REPORTE COMPLETO DE LA SIMULACI脫N DE PLANTA H脥BRIDA")
        print("="*80)
        
        # Tabla de resultados principales
        tabla_resultados = [
            ["馃搮 D铆as simulados", f"{resultados['dias_simulados']}"],
            ["鈿� Generaci贸n total", f"{resultados['generacion_total_mwh']:,.2f} MWh"],
            ["馃挵 Energ铆a vendida", f"{resultados['energia_vendida_mwh']:,.2f} MWh"],
            ["馃挼 Beneficio total", f"${resultados['beneficio_total']:,.2f}"],
            ["馃搱 Beneficio promedio diario", f"${resultados['beneficio_promedio_diario']:,.2f}"],
            ["馃毇 Multas totales", f"${resultados['multas_total']:,.2f}"],
            ["鈿狅笍 Eventos de falla", f"{resultados['eventos_falla']}"],
            ["鉁� Disponibilidad promedio", f"{resultados['disponibilidad_promedio']:.2f}%"],
            ["馃攱 Energ铆a almacenada total", f"{resultados['energia_almacenada_mwh']:,.2f} MWh"],
            ["鉂� Energ铆a desperdiciada", f"{resultados['energia_desperdiciada_mwh']:,.2f} MWh"],
            ["馃攱 Capacidad final BESS", f"{resultados['capacidad_final_bess']:,.2f} MWh"],
            ["馃攱 Nivel final BESS", f"{resultados['nivel_final_bess']:,.2f} MWh"],
            ["馃洟锔� Fuel Oil restante", f"{resultados['fuel_oil_restante']:,.2f} L"]
        ]
        
        print(tabulate(tabla_resultados, headers=["M茅trica", "Valor"], tablefmt="grid"))
        
        if guardar_archivo:
            self._generar_graficos(registro_df)
    
    def _generar_graficos(self, registro_df):
        """
        Genera los gr谩ficos de an谩lisis de la simulaci贸n
        
        Args:
            registro_df (DataFrame): Datos diarios de la simulaci贸n
        """
        print("\n馃搱 Generando gr谩ficos de an谩lisis...")
        
        # Configurar estilo de gr谩ficos
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 10
        
        # Gr谩fico principal: Evoluci贸n temporal
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Generaci贸n vs Demanda
        axes[0,0].plot(registro_df['dia'], registro_df['generacion'], 
                       label='Generaci贸n', linewidth=1.5, color='blue')
        axes[0,0].plot(registro_df['dia'], registro_df['demanda'], 
                       label='Demanda', linewidth=1.5, color='red')
        axes[0,0].set_title('Evoluci贸n de Generaci贸n vs Demanda')
        axes[0,0].set_xlabel('D铆as')
        axes[0,0].set_ylabel('Energ铆a (MWh)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Sistema BESS
        axes[0,1].plot(registro_df['dia'], registro_df['nivel_bess'], 
                       label='Nivel BESS', color='green', linewidth=1.5)
        axes[0,1].plot(registro_df['dia'], registro_df['capacidad_bess'], 
                       label='Capacidad BESS', color='orange', linestyle='--', linewidth=1.5)
        axes[0,1].fill_between(registro_df['dia'], 0, registro_df['nivel_bess'], 
                               alpha=0.3, color='green')
        axes[0,1].set_title('Estado del Sistema de Almacenamiento BESS')
        axes[0,1].set_xlabel('D铆as')
        axes[0,1].set_ylabel('Energ铆a (MWh)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Beneficio acumulado
        beneficio_acumulado = registro_df['beneficio'].cumsum()
        axes[1,0].plot(registro_df['dia'], beneficio_acumulado, 
                       color='darkgreen', linewidth=2)
        axes[1,0].set_title('Evoluci贸n del Beneficio Acumulado')
        axes[1,0].set_xlabel('D铆as')
        axes[1,0].set_ylabel('Beneficio ($)')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].ticklabel_format(style='plain', axis='y')
        
        # 4. Nivel de Fuel Oil
        axes[1,1].plot(registro_df['dia'], registro_df['fuel_oil'], 
                       color='brown', linewidth=1.5)
        axes[1,1].axhline(y=self.MAX_FO * 0.2, color='red', linestyle='--', 
                          label='Nivel cr铆tico (20%)')
        axes[1,1].fill_between(registro_df['dia'], 0, registro_df['fuel_oil'], 
                               alpha=0.3, color='brown')
        axes[1,1].set_title('Nivel de Fuel Oil')
        axes[1,1].set_xlabel('D铆as')
        axes[1,1].set_ylabel('Fuel Oil (L)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simulacion_planta_hibrida.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("鉁� Gr谩ficos guardados como 'simulacion_planta_hibrida.png'")

# ==========================================================================
# FUNCI脫N PRINCIPAL PARA EJECUTAR LA SIMULACI脫N
# ==========================================================================

def main():
    """
    Funci贸n principal que ejecuta la simulaci贸n completa
    """
    print("馃殌 SIMULADOR DE PLANTA H脥BRIDA DE GENERACI脫N ENERG脡TICA")
    print("="*60)
    
    # Crear y configurar la planta
    planta = PlantaHibrida(tiempo_final=365)
    
    # Ejecutar simulaci贸n
    planta.ejecutar_simulacion()
    
    # Generar reporte completo
    planta.generar_reporte(guardar_archivo=True)
    
    # Retornar resultados para an谩lisis adicional
    return planta.obtener_resultados()

# Ejecutar simulaci贸n si el archivo se ejecuta directamente
if __name__ == "__main__":
    resultados, datos_diarios = main()
    print("\n馃幆 Simulaci贸n completada exitosamente!")
    print("馃捑 Los resultados est谩n disponibles en las variables 'resultados' y 'datos_diarios'")