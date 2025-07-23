# 
import bisect
from typing import Optional
import pandas as pd
import numpy as np
import csv
import json
import math
import os
import re
from datetime import datetime
import traceback
from shapely.geometry import Point, shape

class TowerJumpAnalyzer:
    def __init__(self):
        # Configurações principais
        self.min_time_diff_threshold = 5 * 60  # 5 minutos em segundos
        self.max_time_diff_threshold = 60 * 60  # 1 hora em segundos
        self.max_speed_threshold = 900  # max km/h para considerar um tower jump
        self.max_jump_distance = 300  # Distância máxima para considerar um tower jump (em km)
        self.min_jump_distance = 10  # Distância mínima para considerar um tower jump (em km)
        self.date_format = '%m/%d/%y %H:%M'
        self.min_confidence = 0

        # Limiares para resolução de conflitos
        self.min_duration_to_override = 3  # minutos
        self.min_confidence_diff = 20  # porcentagem
        self.min_confidence_absolute = 70  # porcentagem mínima

    def safe_float(self, value):
        """Converte string para float de forma segura"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0

    def haversine_distance_vectorized(self, lon1, lat1, lon2, lat2):
        """
        Calcula a distância entre dois pares de coordenadas (em vetores) usando a fórmula de Haversine.
        Entradas e saídas em quilômetros.
        """

        # Converter graus para radianos
        lon1_rad = np.radians(lon1)
        lat1_rad = np.radians(lat1)
        lon2_rad = np.radians(lon2)
        lat2_rad = np.radians(lat2)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        R = 6371  # Raio da Terra em quilômetros
        return R * c

        
    def is_valid_record(self, record):
        """Verifica se o registro é válido para análise"""
    
        try:          
            if record is None:
                return False
        
            if 'start_time' not in record or 'end_time' not in record:
                return False

            if not isinstance(record['start_time'], datetime) or not isinstance(record['end_time'], datetime):
                return False

            if record['state'] == 'UNKNOWN':
                return False

            if record['latitude'] is None or record['longitude'] is None:
                return False

            if record['latitude'] == 0 and record['longitude'] == 0:
                return False

            if record['latitude'] <= -90 and record['longitude'] <= -180:
                return False
            
            return True
            
        except Exception as e:
            print(record)
            print(f"Erro ao verificar validade do registro: {e}")
            traceback.print_exc()
            return False


    def detect_vehicle_type_vectorized(df):
        """
        Adiciona a coluna 'vehicle_type' ao DataFrame com base em condições vetorizadas
        usando speed, distance e time_diff (em segundos).
        """

        # Converte tempo de segundos para horas
        df['time_hours'] = df['time_diff_seconds'] / 3600
        df['effective_distance'] = df['speed'] * df['time_hours']

        conditions = [
            (df['speed'].between(0, 7)) & (df['time_diff_seconds'] > 60) & (df['effective_distance'] >= df['distance']),
            (df['speed'].between(7, 30)) & (df['time_diff_seconds'] > 60) & (df['effective_distance'] >= df['distance']),
            (df['speed'].between(20, 200)) & (df['time_diff_seconds'] > 60) & (df['effective_distance'] >= df['distance']),
            (df['speed'].between(10, 120)) & (df['time_diff_seconds'] > 60) & (df['effective_distance'] >= df['distance']),
            (df['speed'].between(5, 100)) & (df['time_diff_seconds'] > 60) & (df['effective_distance'] >= df['distance']),
            (df['speed'].between(30, 300)) & (df['time_diff_seconds'] > 60) & (df['effective_distance'] >= df['distance']),
            (df['speed'].between(200, 900)) & (df['time_diff_seconds'] > 60) & (df['effective_distance'] >= df['distance']),
        ]

        choices = ['Andando', 'Bicicleta', 'Carro', 'Ônibus', 'Barco', 'Trem', 'Avião']

        df['vehicle_type'] = np.select(conditions, choices, default='UNKNOWN')

        return df

    def detect_jumps(self, df):
        if df.empty:
            return df

        MAX_TIME_DIFF = 3600

        # 1. Pré-processamento básico
        df = df.sort_values('start_time').reset_index(drop=True)
        df['time_gap'] = (df['start_time'] - df['end_time'].shift(1)).dt.total_seconds().fillna(MAX_TIME_DIFF + 1)
        df['group_break'] = (df['state'] != df['state'].shift(1)) | (df['time_gap'] > MAX_TIME_DIFF)
        df['group_id'] = df['group_break'].cumsum()

        # 2. Criar DataFrame reduzido apenas para grupos
        groups = df.groupby('group_id').agg(
            group_start=('start_time', 'min'),
            group_end=('end_time', 'max'),
            state=('state', 'first'),
            latitude=('latitude', 'first'),
            longitude=('longitude', 'first')
        ).reset_index()

        # 3. Validação de estado (apenas em grupos)
        groups['is_valid'] = (
            (groups['state'] != 'UNKNOWN') &
            groups['latitude'].notna() & (groups['latitude'] != 0) &
            groups['longitude'].notna() & (groups['longitude'] != 0)
        )

        # 4. Rastrear último grupo válido
        groups['prev_valid_state'] = None
        groups['prev_valid_lat'] = None
        groups['prev_valid_lon'] = None
        groups['prev_valid_end'] = pd.NaT
        
        last_valid = {
            'state': None,
            'lat': None,
            'lon': None,
            'end': None
        }

        for i, row in groups.iterrows():
            if row['is_valid']:
                groups.at[i, 'prev_valid_state'] = last_valid['state']
                groups.at[i, 'prev_valid_lat'] = last_valid['lat']
                groups.at[i, 'prev_valid_lon'] = last_valid['lon']
                groups.at[i, 'prev_valid_end'] = last_valid['end']
                
                # Atualizar último válido
                last_valid = {
                    'state': row['state'],
                    'lat': row['latitude'],
                    'lon': row['longitude'],
                    'end': row['group_end']
                }

        # 5. Filtrar apenas grupos válidos para cálculos físicos
        valid_groups = groups[groups['is_valid']].copy()
        valid_groups = valid_groups[valid_groups['prev_valid_state'].notna()]  # Ignorar primeiro válido

        # 6. Cálculos temporais e de movimento (apenas em válidos)
        valid_groups['time_diff'] = (valid_groups['group_start'] - valid_groups['prev_valid_end']).dt.total_seconds()
        
        # 6. Cálculo de distância e velocidade (função Haversine)
        def haversine_vectorized(lon1, lat1, lon2, lat2):
            lon1 = np.radians(lon1.astype(float))
            lat1 = np.radians(lat1.astype(float))
            lon2 = np.radians(lon2.astype(float))
            lat2 = np.radians(lat2.astype(float))
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = np.sin(dlat / 2.0) ** 2 + \
                np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6371 * c
            return km

        valid_groups['distance'] = haversine_vectorized(
            valid_groups['prev_valid_lon'],
            valid_groups['prev_valid_lat'],
            valid_groups['longitude'],
            valid_groups['latitude']
        )
    
        valid_groups['speed'] = valid_groups['distance'] * 3600 / valid_groups['time_diff']
        valid_groups['is_movement_possible'] = (valid_groups['distance'] > 0) & (valid_groups['speed'] < 1000)
        
        # 7. Detecção de anomalias
        valid_groups['state_changed'] = valid_groups['state'] != valid_groups['prev_valid_state']
        valid_groups['valid_state_change_1h'] = valid_groups['state_changed'] & (valid_groups['time_diff'] <= MAX_TIME_DIFF)
        anomaly_candidates = valid_groups[valid_groups['valid_state_change_1h'] & ~valid_groups['is_movement_possible']].copy()

        # Inicializar colunas de anomalia
        valid_groups['tower_jump'] = False
        valid_groups['anomaly_type'] = 'NONE'
        valid_groups['conflict_resolution'] = 'NO_CONFLICT'
        valid_groups['historical_state'] = None
        valid_groups['historical_state_time'] = None

        # Preparar lista de registros válidos para busca histórica
        # Usamos apenas grupos válidos com índice original
        valid_records = []
        for _, row in valid_groups.iterrows():
            valid_records.append((row['group_start'], row.name, row['state']))
        valid_records.sort(key=lambda x: x[0])  # Ordenar por tempo
        valid_times = [vr[0] for vr in valid_records]

        # Busca binária otimizada para cada candidato
        for idx, candidate in anomaly_candidates.iterrows():
            current_time = candidate['group_start']
            base_state = candidate['prev_valid_state']
            current_state = candidate['state']
            
            # Janela temporal: até 1h antes do grupo atual
            low_bound = current_time - pd.Timedelta(seconds=MAX_TIME_DIFF)
            
            # Encontrar índices na lista ordenada
            left_idx = bisect.bisect_left(valid_times, low_bound)
            right_idx = bisect.bisect_right(valid_times, current_time) - 1
            
            historical_match = None
            
            # Procurar do mais recente para o mais antigo
            if left_idx <= right_idx:
                for pos in range(right_idx, left_idx - 1, -1):
                    record_time, record_idx, record_state = valid_records[pos]
                    
                    # Ignorar o próprio grupo e grupos posteriores
                    if record_idx >= idx:
                        continue
                        
                    # Critério de compatibilidade:
                    # 1. Estado diferente do estado anterior (base_state)
                    # 2. Mesmo estado do grupo atual (current_state)
                    if record_state != base_state and record_state == current_state:
                        historical_match = (record_idx, record_state, record_time)
                        break
            
            # Classificação de anomalias
            if historical_match is None:
                valid_groups.at[idx, 'tower_jump'] = True
                valid_groups.at[idx, 'anomaly_type'] = 'NO_HISTORICAL_SUPPORT'
            else:
                rec_idx, hist_state, hist_time = historical_match
                valid_groups.at[idx, 'tower_jump'] = True
                valid_groups.at[idx, 'anomaly_type'] = 'HISTORICAL_CONFLICT'
                valid_groups.at[idx, 'historical_state'] = hist_state
                valid_groups.at[idx, 'historical_state_time'] = hist_time
                
                # Verificar recuperação automática
                next_idx = idx + 1
                if next_idx in valid_groups.index and valid_groups.loc[next_idx, 'state'] == base_state:
                    valid_groups.at[idx, 'conflict_resolution'] = 'AUTO_RECOVERY'
                
        # 8. Mesclar resultados de volta ao DataFrame original
        df = df.merge(
            groups[['group_id', 'prev_valid_state', 'prev_valid_lat', 'prev_valid_lon', 'prev_valid_end']],
            on='group_id',
            how='left'
        )
        
        df = df.merge(
            valid_groups[['group_id', 'distance', 'speed', 'is_movement_possible', 'anomaly_type', 'tower_jump']],
            on='group_id',
            how='left'
        ).fillna({'distance': 0, 'speed': 0, 'is_movement_possible': True, 'tower_jump': False})
        
        # 9. Preencher colunas derivadas restantes
        df['physical_consistency'] = np.where(
            df['tower_jump'],
            'INCONSISTENT',
            'CONSISTENT'
        )
        
        return df
 
    def load_data(self, file_path):
        """Carrega e processa os dados do arquivo CSV usando pandas"""
        try:
            # Carrega o CSV com pandas
            df = pd.read_csv(file_path, parse_dates=['UTCDateTime'], date_format='%m/%d/%y %H:%M')
            
            required_columns = {'UTCDateTime', 'LocalDateTime', 'State', 'Latitude', 'Longitude'}
            if not all(col in df.columns for col in required_columns):
                missing = required_columns - set(df.columns)
                print(f"Erro: Colunas obrigatórias faltando: {missing}")
                return pd.DataFrame()

            df[['State']] = df[['State']].fillna('UNKNOWN')
            df[['Latitude','Longitude']] =  df[['Latitude','Longitude']].fillna('0.0')

            # Processa os dados
            processed_data = []
           
            for index, row in df.iterrows():
                try:
                    start_time = row.get('UTCDateTime')

                    # Processa latitude e longitude
                    unformatted_lat = row.get('Latitude')
                    if unformatted_lat is not None and isinstance(unformatted_lat, str):
                        unformatted_lat = unformatted_lat[:8]
                    
                    unformatted_lon = row.get('Longitude')
                    if isinstance(unformatted_lon, str):
                        unformatted_lon = unformatted_lon[:9]
                    
                    lat = self.safe_float(unformatted_lat)
                    lon = self.safe_float(unformatted_lon)
                    
                    # Determina o estado
                    state = row.get('State', 'UNKNOWN')

                    # Cria a entrada de dados
                    entry = {
                        'start_time': start_time,
                        'end_time': start_time,
                        'latitude': lat,
                        'longitude': lon,
                        'state': state,                     
                    }
                    
                    processed_data.append(entry)
                
                except Exception as e:
                    print(f"Erro ao processar linha {index+1}: {str(e)}")
                    continue
            
            # Cria DataFrame com os dados processados
            result_df = pd.DataFrame(processed_data)
            
            if result_df.empty:
                print("Aviso: O arquivo foi lido, mas nenhum dado válido foi encontrado")
                return pd.DataFrame()
            
            # Ordena por tempo
            result_df = result_df.sort_values('start_time')
            
            print(f"Carregados {len(result_df)} registros válidos")
            
            unknown_count = sum(1 for state in result_df['state'] if state == 'UNKNOWN')
            if unknown_count > 0:
                print(f"AVISO: {unknown_count} registros ({unknown_count/len(result_df):.2%}) com estado UNKNOWN")
                if unknown_count/len(result_df) > 0.05:
                    print("  Ação recomendada: Verificar qualidade das coordenadas e mapeamento de estados")
            
            return result_df
        
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Erro inesperado ao ler arquivo: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()

    def format_distance(self, distance):
        """Formata a distância em km com 2 casas decimais"""
        try:
            if distance is None or np.isnan(distance):
                return ''
            return f"{distance:.2f} km" if distance >= 0 else ''
        except (ValueError, TypeError) as e:
            print(f"Erro ao formatar distância: {str(e)}")
            traceback.print_exc()
            return ''

    def format_speed(self, speed):
        """Formata a velocidade em km/h com 2 casas decimais"""
        if speed is None or np.isnan(speed):
            return ''
        return f"{speed:.2f} km/h" if speed >= 0 else ''
    
    import traceback

    def generate_report(self, df, output_file):
        """Gera um relatório CSV com todas as colunas processadas"""
        if df.empty:
            print("Nenhum dado para gerar relatório")
            return False

        try:
            # 1. Formatar todas as colunas de data/hora
            datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
            for col in datetime_cols:
                df[f'{col}_str'] = df[col].dt.strftime(self.date_format)
            
            # 2. Formatar colunas numéricas específicas
            df['speed_str'] = df['speed'].apply(self.format_speed)
            df['distance_str'] = df['distance'].apply(self.format_distance)
            
            # 3. Preparar lista de todas as colunas para exportação
            all_columns = list(df.columns)
            
            # 4. Reordenar colunas para priorizar informações principais
            priority_columns = [
                'start_time_str', 'end_time_str', 'group_start_str', 'group_end_str',
                'state', 'anomaly_summary', 'tower_jump', 'physical_consistency',
                'conflict_resolution', 'anomaly_type', 'prev_group_state', 'historical_state', 'latitude', 'longitude', 'group_lat', 'group_lon',
                'speed_str', 'distance_str', 'is_movement_possible'
            ]
            
            # 5. Criar lista ordenada de colunas (prioritárias + demais)
            export_columns = [col for col in priority_columns if col in all_columns]
            remaining_columns = [col for col in all_columns if col not in priority_columns and not col.endswith('_str')]
            export_columns += sorted(remaining_columns)
            
            # 6. Preparar DataFrame para exportação
            export_df = df[export_columns]
            
            # 7. Renomear colunas formatadas para nomes amigáveis
            rename_map = {
                'start_time_str': 'start_time',
                'end_time_str': 'end_time',
                'group_start_str': 'group_start',
                'group_end_str': 'group_end',
                'speed_str': 'speed',
                'distance_str': 'distance',
                'prev_group_state': 'previous_state',
                'group_lat': 'group_latitude',
                'group_lon': 'group_longitude'
            }
            export_df = export_df.rename(columns=rename_map)
            
            # 8. Exportar para CSV
            export_df.to_csv(output_file, index=False)
            print(f"Relatório gerado com sucesso: {output_file}")
            print(f"Total de colunas exportadas: {len(export_columns)}")
            print(f"Total de registros: {len(export_df)}")
            return True
            
        except Exception as e:
            print(f"Erro ao gerar relatório: {str(e)}")
            traceback.print_exc()
            return False

    # def generate_report(self, df, output_file):
    #     """Gera um relatório CSV com os dados processados"""
    #     if df.empty:
    #         print("Nenhum dado para gerar relatório")
    #         return False

    #     try:
    #         # Formata as colunas de data/hora
    #         df['start_time_str'] = df['start_time'].dt.strftime(self.date_format)
    #         df['end_time_str'] = df['end_time'].dt.strftime(self.date_format)
            
    #         # Formata as colunas numéricas
    #         df['speed_str'] = df['speed'].apply(self.format_speed)
    #         df['distance_str'] = df['distance'].apply(self.format_distance)

    #         # Prepara o DataFrame para exportação
    #         export_df = df[[
    #             'start_time_str', 'end_time_str', 'latitude', 'longitude',
    #             'state', 'discarded_state',
    #             'duration', 'speed_str', 'distance_str', 'vehicle_type', 
    #             'movement_possible', 'tower_jump', 'conflict_resolution', 
    #             'resolved_by', 'location_score', 'confidence'
    #         ]]
            
    #         # Renomeia as colunas
    #         export_df = export_df.rename(columns={
    #             'start_time_str': 'start_time',
    #             'end_time_str': 'end_time',
    #             'speed_str': 'speed',
    #             'distance_str': 'distance'
    #         })
            
    #         # Exporta para CSV
    #         export_df.to_csv(output_file, index=False)
    #         return True
            
    #     except Exception as e:
    #         print(f"Erro ao gerar relatório: {str(e)}")
    #         traceback.print_exc()
    #         return False

    def print_summary(self, df):
        """Imprime um resumo dos dados processados"""
        if df.empty:
            print("Nenhum dado para resumo")
            return

        total_records = len(df)
        tower_jump_count = df['tower_jump'].sum()
        unknown_state_count = df['state'].eq('UNKNOWN').sum()

        print("\n=== RESUMO DOS DADOS ===")
        print(f"Total de registros: {total_records}")
        print(f"Registros com Tower Jump: {tower_jump_count} ({tower_jump_count/total_records:.2%})")
        print(f"Registros com estado UNKNOWN: {unknown_state_count} ({unknown_state_count/total_records:.2%})")

def main():
    print("=== Tower Jump Analyzer ===")
    print("Sistema avançado de análise de localização e detecção de tower jumps usando pandas\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "localization")
    input_file = os.path.join(data_dir, "CarrierData.csv")
    output_file_name = 'TowerJumpReport.csv'
    # output_file_name = 'main_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    output_file = os.path.join(base_dir, output_file_name)

    if not os.path.exists(input_file):
        print(f"Erro: Arquivo de entrada não encontrado em {input_file}")
        print("Certifique-se que o arquivo CarrierData.csv está na pasta 'localization'")
        return

    analyzer = TowerJumpAnalyzer()
    print("Carregando e processando dados...")
    df = analyzer.load_data(input_file)

    if df.empty:
        print("Nenhum dado válido encontrado no arquivo de entrada.")
        return

    print("Detectando tower jumps...")
    df = analyzer.detect_jumps(df)

    # print("Resolvendo conflitos de estado...")
    # df = analyzer.resolve_state_conflicts(df)

    # print("Obtendo localização atual...")
    # current_location = analyzer.get_current_location(df)

    print("Gerando relatório...")
    if analyzer.generate_report(df, output_file):
        print(f"\nRelatório gerado com sucesso em: {output_file}")
    else:
        print("Falha ao gerar relatório.")
        return

    analyzer.print_summary(df)

if __name__ == "__main__":
    main()