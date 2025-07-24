# 
import bisect
import pandas as pd
import numpy as np
import os
import traceback

class TowerJumpAnalyzer:
    def __init__(self):
        self.date_format = '%m/%d/%y %H:%M'

    def safe_float(self, value):
        """Safely converts string to float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0

    def is_movement_possible(self, dist_km, time_sec, max_speed_kmh=300):
        """Verifica se o deslocamento entre dois pontos é possível fisicamente."""
        max_speed_mps = (max_speed_kmh * 1000) / 3600
        time_sec = time_sec.replace(0, np.nan)  # evitar divisão por zero
        speed_mps = (dist_km * 1000) / time_sec
        return speed_mps <= max_speed_mps

    
    def haversine_vectorized(self, lon1, lat1, lon2, lat2):
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

    def _preprocess_dataframe(self, df):
        """Ordena, preenche campos e marca registros válidos."""
        df = df.sort_values('start_time').reset_index(drop=True)
        df['state'] = df['state'].fillna('')
        df['is_valid'] = (
            (df['state'] != '') &
            df['latitude'].notna() & (df['latitude'] != 0) &
            df['longitude'].notna() & (df['longitude'] != 0)
        )
        df['valid_state'] = df['state'].where(df['is_valid'])
        df['fwd_valid_state'] = df['valid_state'].ffill()
        return df

    def _assign_group_ids(self, df):
        """Atribui IDs de grupo apenas para registros válidos com mudança de estado."""
        df['prev_valid_state'] = df['valid_state'].ffill().shift()
        df['state_change'] = (
            df['valid_state'].notna() & (df['valid_state'] != df['prev_valid_state'])
        )

        df['group_id'] = df['state_change'].cumsum()
        df.loc[~df['is_valid'], 'group_id'] = pd.NA

        return df

    def _analyze_valid_groups(self, df):
        """Agrupa registros válidos, calcula distância, tempo e marca tower jumps."""
        valid = df[df['group_id'].notna()].copy()

        group_info = valid.groupby('group_id').agg({
            'valid_state': 'first',
            'latitude': 'first',
            'longitude': 'first',
            'start_time': 'first',
            'end_time': 'last'
        }).reset_index()

        group_info = group_info.rename(columns={
            'start_time': 'group_start_time',
            'end_time': 'group_end_time'
        })

        group_info['duration'] = (
            group_info['group_end_time'] - group_info['group_start_time']
        ).dt.total_seconds()

        group_info['prev_state'] = group_info['valid_state'].shift()
        group_info['tower_jump'] = np.where(
            (group_info['valid_state'] != group_info['prev_state']) &
            (group_info['group_id'] != group_info['group_id'].min()),
            True,
            pd.NA
        )

        group_info['time_diff'] = (
            group_info['group_start_time'] - group_info['group_end_time'].shift()
        ).dt.total_seconds()

        group_info['distance_km'] = self.haversine_vectorized(
            group_info['longitude'].shift(), group_info['latitude'].shift(),
            group_info['longitude'], group_info['latitude']
        )

        return df, group_info

    def _map_group_info(self, df, group_info):
        """Mapeia informações do grupo de volta para o dataframe original."""
        map_cols = ['tower_jump', 'group_start_time', 'group_end_time', 'duration',
                    'distance_km', 'time_diff', 'confidence', 'confidence_display']

        for col in map_cols:
            df[col] = df['group_id'].map(group_info.set_index('group_id')[col])

        return df
    
    def detect_jumps_novo(self, df):
        """Detecta e agrupa estados válidos consecutivos, calculando tempo, distância, 
        duração e tower jumps com confiança."""
        if df.empty:
            return df

        df = self._preprocess_dataframe(df)
        df = self._assign_group_ids(df)
        df, group_info = self._analyze_valid_groups(df)

        group_info['is_movement_possible'] = self.is_movement_possible(
            group_info['distance_km'], group_info['time_diff']
        )

        group_info = self.calculate_confidence(group_info)
        df = self._map_group_info(df, group_info)

        return df

    def calculate_confidence(self, groups):
        """Calcula a confiança na detecção de tower jumps."""
        groups['confidence'] = 0.0

        cond_jump = groups['tower_jump'] == True
        groups_jump = groups[cond_jump].copy()

        base_score = 0.0
        movement_bonus = 0.30
        distance_bonus = 0.20
        duration_bonus = 0.10

        score = (
            movement_bonus * groups_jump['is_movement_possible'].fillna(False).astype(float) +
            distance_bonus * (groups_jump['distance_km'] > 100).astype(float) +
            duration_bonus * (groups_jump['duration'] < 600).astype(float)
        )

        groups.loc[cond_jump, 'confidence'] = (score.clip(0, 1.0) * 100).round(1)
        groups['confidence_display'] = groups['confidence'].astype(str) + '%'

        return groups

    def load_data(self, file_path):
        """ Loads and process data to pandas dataframe """
        try:
            df = pd.read_csv(file_path, parse_dates=['UTCDateTime'], date_format='%m/%d/%y %H:%M')
            
            required_columns = {'UTCDateTime', 'LocalDateTime', 'State', 'Latitude', 'Longitude'}
            if not all(col in df.columns for col in required_columns):
                missing = required_columns - set(df.columns)
                print(f"Erro: Colunas obrigatórias faltando: {missing}")
                return pd.DataFrame()

            df[['State']] = df[['State']]
            df[['Latitude','Longitude']] =  df[['Latitude','Longitude']].fillna('0.0')

            processed_data = []
           
            for index, row in df.iterrows():
                try:
                    start_time = row.get('UTCDateTime')

                    unformatted_lat = row.get('Latitude')
                    if unformatted_lat is not None and isinstance(unformatted_lat, str):
                        unformatted_lat = unformatted_lat[:8]
                    
                    unformatted_lon = row.get('Longitude')
                    if isinstance(unformatted_lon, str):
                        unformatted_lon = unformatted_lon[:9]
                    
                    lat = self.safe_float(unformatted_lat)
                    lon = self.safe_float(unformatted_lon)
                    
                    state = row.get('State', '')

                    entry = {
                        'start_time': start_time,
                        'end_time': start_time,
                        'latitude': lat,
                        'longitude': lon,
                        'state': state,                     
                    }
                    
                    processed_data.append(entry)
                
                except Exception as e:
                    print(f"Error at line {index+1}: {str(e)}")
                    continue
            
            result_df = pd.DataFrame(processed_data)
            
            if result_df.empty:
                print("Warning: No valid data found.")
                return pd.DataFrame()
            
            result_df = result_df.sort_values('start_time')
            
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
            # df['speed_str'] = df['speed'].apply(self.format_speed)
            # df['distance_str'] = df['distance'].apply(self.format_distance)
            
            # 3. Preparar lista de todas as colunas para exportação
            all_columns = list(df.columns)
            
            # 4. Reordenar colunas para priorizar informações principais
            priority_columns = [
                'start_time_str', 'end_time_str', 'is_valid', 'state', 'valid_state', 
                'state_change', 'prev_valid_state', 'fwd_valid_state', 'tower_jump',
                'group_id', 'group_start_time', 'group_end_time', 'duration'
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
            export_df.to_csv(output_file, index=False,lineterminator="")
            print(f"Relatório gerado com sucesso: {output_file}")
            print(f"Total de colunas exportadas: {len(export_columns)}")
            print(f"Total de registros: {len(export_df)}")

            return True
            
        except Exception as e:
            print(f"Erro ao gerar relatório: {str(e)}")
            traceback.print_exc()
            return False

    def print_summary(self, df):
        """Imprime um resumo dos dados processados"""
        if df.empty:
            print("Nenhum dado para resumo")
            return

        total_records = len(df)
        tower_jump_count = df['tower_jump'].sum()
        unknown_state_count = df['state'].eq('').sum()

        print("\n=== RESUMO DOS DADOS ===")
        print(f"Total de registros: {total_records}")
        print(f"Registros com Tower Jump: {tower_jump_count} ({tower_jump_count/total_records:.2%})")
        print(f"Registros com estado vazio: {unknown_state_count} ({unknown_state_count/total_records:.2%})")

def main():
    print("=== Tower Jump Analyzer ===")
    print("Sistema avançado de análise de localização e detecção de tower jumps usando pandas\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "localization")
    input_file = os.path.join(data_dir, "CarrierData.csv")
    output_file_name = 'TowerJumpReport.csv'
    output_file = os.path.join(base_dir, output_file_name)
    output_file2 = os.path.join(base_dir, 'result.csv')

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
    df = analyzer.detect_jumps_novo(df)

    print("Gerando relatório...")
    if analyzer.generate_report(df, output_file):
        print(f"\nRelatório gerado com sucesso em: {output_file}")
    else:
        print("Falha ao gerar relatório.")
        return

    analyzer.print_summary(df)

if __name__ == "__main__":
    main()