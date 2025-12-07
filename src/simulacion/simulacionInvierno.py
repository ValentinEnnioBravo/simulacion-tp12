"""
SIMULACIÓN DE PLANTA DE GENERACIÓN - INVIERNO

Implementación basada en el diagrama de flujo provisto. Esta versión reutiliza
datos de los CSVs en `public/` cuando están disponibles y aplica una lógica
de ola fría característica de invierno (aumenta la demanda en días puntuales).

Ejecutar: `python3 src/simulacion/simulacionInvierno.py`
"""
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib


class SimulacionInvierno:
    def __init__(self):
        # Tiempo y eventos
        self.T = 0
        self.TF = 365
        self.RP = float('inf')   # llegada de repuesto (infinito hasta pedido)
        self.RFO = float('inf')  # llegada camión fuel oil

        # Turbina vapor
        self.H = 1  # 1 = habilitada, 0 = deshabilitada

        # Fuel oil
        self.MAX_FO = 500000.0
        self.FO = self.MAX_FO
        self.PFO = 200000.0

        # Generación y BESS
        self.GD = 0.0
        self.TG = 'ESTANDAR'  # Tipo de guardia: 'ESTANDAR' o 'MINIMA'
        self.BESS = 0.0
        self.CBESS = 200.0
        self.CCC = 0  # contador ciclos de carga

        # Economía
        self.BENEF = 0.0
        self.PV = 87.0   # precio venta $/MWh
        self.CC = 30.0   # costo combustible $/MWh
        self.CU = 5.0    # costo uso
        self.PM = 180.0   # precio multa por MWh no satisfecho
        self.I = 1.0

        # Estadísticas
        self.STB = 0.0
        self.SAB = 0.0
        self.STM = 0.0
        self.C = 0
        self.STI = 0.0

        # Parámetros climáticos/invernales
        # Probabilidad de ola fría que aumenta demanda
        self.prob_ola_frio = 0.25
        self.factor_demanda_ola_frio = 0.15  # +15% demanda

        # Cargar datos desde CSVs (si están disponibles)
        self.base_path = os.path.join(os.path.dirname(__file__), '..', '..')
        self.public_path = os.path.join(self.base_path, 'public')
        self._cargar_datos()

        # Resultados diarios para muestreo
        self.resultados = []
        # Series temporales diarias (una entrada por día)
        self.timeseries = []

    def _cargar_datos(self):
        """Carga CSVs si existen y prepara vectores de muestreo."""
        try:
            dem2_path = os.path.join(self.public_path, 'Demanda_Segundo_Semestre.csv')
            self.df_dem2 = pd.read_csv(dem2_path)
            # columna con demanda en MWh
            if 'Demanda_MWh' in self.df_dem2.columns:
                #! OJO ACA METI UNA NEGRADA
                self.demanda_samples = self.df_dem2['Demanda_MWh'].values * 0.45
            else:
                self.demanda_samples = np.full(1000, 1650.0)

        except Exception:
            self.demanda_samples = np.full(1000, 1650.0)

        try:
            cc1_path = os.path.join(self.public_path, 'Generacion_CC1.csv')
            self.df_cc1 = pd.read_csv(cc1_path)
            self.cc1_samples = self.df_cc1['Generacion_MW'].values if 'Generacion_MW' in self.df_cc1.columns else np.full(1000, 700.0)
        except Exception:
            self.cc1_samples = np.full(1000, 700.0)

        try:
            cc2_path = os.path.join(self.public_path, 'Generacion_CC2.csv')
            self.df_cc2 = pd.read_csv(cc2_path)
            self.cc2_samples = self.df_cc2['Generacion_MW'].values if 'Generacion_MW' in self.df_cc2.columns else np.full(1000, 600.0)
        except Exception:
            self.cc2_samples = np.full(1000, 600.0)

        try:
            tv_path = os.path.join(self.public_path, 'Generacion_TV.csv')
            self.df_tv = pd.read_csv(tv_path)
            self.tv_samples = self.df_tv['Generacion_MW'].values if 'Generacion_MW' in self.df_tv.columns else np.full(1000, 400.0)
        except Exception:
            self.tv_samples = np.full(1000, 400.0)

        try:
            ad_path = os.path.join(self.public_path, 'Autodescarga_BESS.csv')
            self.df_ad = pd.read_csv(ad_path)
            if 'Autodescarga_Porcentaje' in self.df_ad.columns:
                self.ad_samples = self.df_ad['Autodescarga_Porcentaje'].values
            else:
                self.ad_samples = np.full(1000, 0.02)
        except Exception:
            self.ad_samples = np.full(1000, 0.02)

    # Muestreos
    def sample_demanda(self):
        return float(np.random.choice(self.demanda_samples))

    def sample_cc1(self):
        return float(np.random.choice(self.cc1_samples))

    def sample_cc2(self):
        return float(np.random.choice(self.cc2_samples))

    def sample_tv(self):
        return float(np.random.choice(self.tv_samples))

    def sample_ad(self):
        return float(np.random.choice(self.ad_samples))

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
        GDCC1 = self.sample_cc1()
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

        # GDCC2
        GDCC2 = self.sample_cc2()

        # 4. Rama principal: gas o fuel oil
        if gas:
            # operación a gas
            r = random.random()
            if self.TG == 'ESTANDAR':
                if r <= 0.03:
                    self.GD += GDCC2 * 0.73
                else:
                    self.GD += GDCC2
            else:
                if r <= 0.045:
                    self.GD += GDCC2 * 0.73
                else:
                    self.GD += GDCC2

            # En invierno no aplicamos penalización por ola de calor

            # Turbina a vapor solo si H == 1
            if self.H == 1:
                GDTV = self.sample_tv()
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

        else:
            # operación fuel oil
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
                self.GD += GDCC2 * 0.73
            else:
                self.GD += GDCC2

            # Reposición si bajo
            if self.FO <= 2500.0:
                self.RFO = self.T + 2

            # Costos fijos operación fuel oil
            self.BENEF -= 15000.0

        # 5. Costos variables de combustible
        self.BENEF -= (self.CC * self.GD)

        # 6. Balance energía y BESS
        DV = self.sample_demanda()

        # Aplicar efecto de ola fría en invierno (aumenta la demanda)
        if random.random() < self.prob_ola_frio:
            DV = DV * (1 + self.factor_demanda_ola_frio)

        if self.GD >= DV:
            # Exceso
            self.BENEF += (self.PV * DV)
            exceso = self.GD - DV
            self.BESS += exceso

            if self.BESS > self.CBESS:
                # saturado
                sobrante_almacenado = self.CBESS - (self.BESS - exceso)
                self.BESS = self.CBESS
                self.STB += sobrante_almacenado
                self.CCC += 1
            else:
                self.STB += (self.PV - self.CU)

        else:
            # Déficit
            self.BENEF += (self.PV * self.GD)
            demanda_restante = DV - self.GD
            #! self.GD = 0.0

            if self.BESS >= demanda_restante:
                self.SAB += demanda_restante
                self.BESS -= demanda_restante
                self.BENEF += demanda_restante * self.PV
            else:
                # batería insuficiente
                self.BENEF += (self.BESS * self.PV)
                demanda_restante -= self.BESS
                self.BESS = 0.0
                # multas por déficit
                self.BENEF -= (demanda_restante * self.PM)
                self.STM += (demanda_restante * self.PM)
                self.C += 1

        # 7. Mantenimiento baterías por ciclos
        if self.CCC == 100:
            self.CBESS = self.CBESS * (1 - 0.01)
            self.CCC = 0

        # 8. Cierre del día: autodescarga y amortización
        AD = self.sample_ad()
        self.BESS = self.BESS * (1 - AD)

        # Amortización BESS
        self.BENEF -= self.CABESS()

        # Registro STI (ingresos turbina)
        # aproximamos como ingreso proporcional al mínimo entre demanda y generación TV si H==1
        if self.H == 1:
            # calculamos aporte aproximado TV (si existe en el día)
            # para simplificar, aproximamos con una muestra
            gd_tv = self.sample_tv()
            self.STI += (self.PV - self.CU) * min(DV, gd_tv) / self.I

        # 9. Falla catastrófica turbina vapor
        if random.random() <= 0.01 and self.H != 0:
            self.RP = self.T + 3
            self.H = 0

        # Guardar muestreo del día
        # Registrar serie diaria completa
        self.timeseries.append({
            'Dia': self.T,
            'Generacion_Total': float(self.GD),
            'Demanda': float(DV),
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

    def generar_graficos(self, output_dir='results', show=True):
        """Genera y guarda gráficos útiles para la toma de decisiones.

        - Beneficio acumulado vs día
        - Nivel BESS vs día
        - Generación total vs Demanda vs día
        - Nivel Fuel Oil vs día
        - Histograma del cambio diario de beneficio
        """
        if not self.timeseries:
            print('No hay datos para graficar.')
            return {}

        out_path = pathlib.Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.timeseries)
        df = df.sort_values('Dia')

        dias = df['Dia'].values

        # Beneficio acumulado
        plt.figure()
        plt.plot(dias, df['Beneficio_Acumulado'].values, label='Beneficio acumulado')
        plt.xlabel('Día')
        plt.ylabel('Beneficio ($)')
        plt.title('Beneficio acumulado vs Día')
        plt.grid(True)
        plt.legend()
        p1 = out_path / 'beneficio_acumulado.png'
        plt.savefig(p1)

        # BESS nivel
        plt.figure()
        plt.plot(dias, df['BESS_Nivel'].values, label='BESS nivel', color='orange')
        plt.xlabel('Día')
        plt.ylabel('Nivel BESS (MWh)')
        plt.title('Nivel de BESS vs Día')
        plt.grid(True)
        plt.legend()
        p2 = out_path / 'bess_nivel.png'
        plt.savefig(p2)

        # Generación vs Demanda
        plt.figure()
        plt.plot(dias, df['Generacion_Total'].values, label='Generación total', color='green')
        plt.plot(dias, df['Demanda'].values, label='Demanda', color='red', alpha=0.7)
        plt.xlabel('Día')
        plt.ylabel('MWh')
        plt.title('Generación total y Demanda vs Día')
        plt.grid(True)
        plt.legend()
        p3 = out_path / 'generacion_vs_demanda.png'
        plt.savefig(p3)

        # Fuel Oil nivel
        plt.figure()
        plt.plot(dias, df['FO'].values, label='Fuel Oil (FO)', color='brown')
        plt.xlabel('Día')
        plt.ylabel('FO (unidades)')
        plt.title('Nivel de Fuel Oil vs Día')
        plt.grid(True)
        plt.legend()
        p4 = out_path / 'fuel_oil_nivel.png'
        plt.savefig(p4)

        # Histograma del cambio diario de beneficio
        beneficio = df['Beneficio_Acumulado'].values
        cambios = np.diff(beneficio)
        plt.figure()
        plt.hist(cambios, bins=40, color='purple', edgecolor='k', alpha=0.7)
        plt.xlabel('Cambio diario beneficio ($)')
        plt.ylabel('Frecuencia')
        plt.title('Histograma: cambio diario de beneficio')
        p5 = out_path / 'hist_cambio_beneficio.png'
        plt.savefig(p5)

        # Mostrar todas las figuras simultáneamente si se pidió
        if show:
            plt.show()
        # Cerrar todas las figuras después de mostrarlas
        plt.close('all')

        print(f'Graficos guardados en {out_path.resolve()}')
        return {
            'beneficio_acumulado': str(p1),
            'bess_nivel': str(p2),
            'generacion_vs_demanda': str(p3),
            'fuel_oil_nivel': str(p4),
            'hist_cambio_beneficio': str(p5)
        }

    def CABESS(self):
        # costo amortización diario del BESS (valor de ejemplo)
        return 5000.0

    def ejecutar(self, dias=None, tipo_guardia=None, max_fo=None, pfo=None, cbess=None, pv=None, cc=None, pm=None):
        # 1. Días
        if dias is not None:
            self.TF = int(dias)
        else:
            try:
                entrada = input('Ingrese número de días a simular (por defecto 365): ').strip()
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
                mfo = input('Ingrese capacidad máxima de Fuel Oil [500000.0]: ').strip()
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

        # 4. Precio venta
        if pv is not None:
            self.PV = float(pv)
        else:
            try:
                pv_in = input('Ingrese precio de venta $/MWh [87.0]: ').strip()
                if pv_in:
                    self.PV = float(pv_in)
            except Exception:
                self.PV = 50.0

        # 5. Costo combustible
        if cc is not None:
            self.CC = float(cc)
        else:
            try:
                cc_in = input('Ingrese costo combustible $/MWh [30.0]: ').strip()
                if cc_in:
                    self.CC = float(cc_in)
            except Exception:
                self.CC = 10.0

        # 6. Precio multa
        if pm is not None:
            self.PM = float(pm)
        else:
            try:
                pm_in = input('Ingrese precio multa por MWh no satisfecho [180.0]: ').strip()
                if pm_in:
                    self.PM = float(pm_in)
            except Exception:
                self.PM = 80.0

        print(f"Iniciando simulación invierno {self.TF} días (TG={self.TG})...")
        while self.T < self.TF:
            self.simular_turno()

        print('Simulación finalizada')
        return self.resumen()

    def resumen(self):
        return {
            'beneficio_total': self.BENEF,
            'capacidad_final_bess': self.BESS,
            'capacidad_maxima_bess': self.CBESS,
            'energia_sobrante_total': self.STB,
            'ingresos_turbina_promedio': self.STI / max(1, self.T),
            'dias_simulados': self.T,
            'resultados_muestreados': self.resultados
        }


if __name__ == '__main__':
    sim = SimulacionInvierno()
    resultados = sim.ejecutar()
    print('\n--- Resumen ---')
    for k, v in resultados.items():
        if k != 'resultados_muestreados':
            print(f"{k}: {v}")
    # imprimir primeros registros muestreados
    if resultados['resultados_muestreados']:
        import pprint
        print('\nMuestreo días:')
        #! pprint.pprint(resultados['resultados_muestreados'][:5])
        pprint.pprint(resultados['resultados_muestreados'])
    # Generar y mostrar gráficos
    sim.generar_graficos(show=True)