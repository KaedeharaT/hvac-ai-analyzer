"""Create clearly synthetic import compatibility fixtures outside research data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def create(destination: Path) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    def entry(name, path, **extra):
        entries.append({'name': 'SYNTHETIC TEST ' + name, 'path': str(path.resolve()), **extra})
    frame = pd.DataFrame({'时间': pd.date_range('2025-01-01', periods=48, freq='30min'),
                          'CH-902 供水温度 (°C)': [6.] * 48, 'CH-902 回水温度 (°C)': [11.] * 48,
                          'CH-902 流量 (m3/h)': [18.] * 48, 'CH-902 功率 (kW)': [30.] * 48})
    frame.loc[2, '时间'] = pd.NaT; frame.loc[5, '时间'] = frame.loc[4, '时间']
    frame.iloc[10, 2] = None; frame.iloc[15, 4] = 0
    path = destination / 'synthetic-chinese.csv'; frame.to_csv(path, index=False)
    entry('Chinese missing/duplicate values', path, expect_cop=True, reviews=[
        {'column': 'CH-902 供水温度 (°C)', 'label': 'heat_source_supply_temp', 'equipment': 'CH-902', 'unit': '°C'}])
    english = pd.DataFrame({'Date': ['2025-02-01'] * 24, 'Time': [f'{i:02}:00' for i in range(24)],
                            'CH-903 Power W': [10000] * 24, 'CH-903 Water Flow L/s': [1.] * 24,
                            'CH-903 LWT (°C)': [6.] * 24, 'CH-903 EWT (°C)': [11.] * 24})
    path = destination / 'synthetic-multiple-sheets.xlsx'
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({'Notes': ['SYNTHETIC TEST DATA; no research observations']}).to_excel(writer, sheet_name='Readme', index=False)
        english.to_excel(writer, sheet_name='Measurements', index=False)
    entry('English separate date/clock, W, L/s', path, sheet='Measurements', expect_cop=True)
    path = destination / 'synthetic-cp932.csv'
    path.write_text('SYNTHETIC TEST DATA\n時刻;HP-904 往水温度℃;HP-904 還水温度℃;電力 kW;電力 kW\n2025/03/01 00:00;6;11;10;12\n2025/03/01 01:00;6;11;10;12\n', encoding='cp932')
    entry('Japanese CP932 preamble duplicate headers', path, no_cop=True)
    path = destination / 'synthetic-unknown-units.csv'
    pd.DataFrame({'timestamp': pd.date_range('2025-04-01', periods=8, freq='h'),
                  'CH-905 LWT': [6.] * 8, 'CH-905 EWT': [11.] * 8,
                  'CH-905 Flow': [60.] * 8, 'CH-905 Power': [10.] * 8}).to_csv(path, index=False)
    entry('Unconfirmed units', path, no_cop=True)
    path = destination / 'synthetic-mixed-no-power.csv'
    pd.DataFrame({'CH-906 流量 (m3/h)': [18.] * 8, '日期': pd.date_range('2025-05-01', periods=8, freq='h'),
                  'CH-906 往水温度 (°C)': [6.] * 8, 'CH-906 Return Water Temperature (°C)': [11.] * 8}).to_csv(path, index=False, encoding='utf-16')
    entry('Mixed language UTF16 reordered missing power', path, no_cop=True)
    path = destination / 'synthetic-gb18030.csv'
    pd.DataFrame({'时间': ['2025年05月01日 00:00', '2025年05月01日 01:00'], '室外温度 °C': [20., 22.]}).to_csv(path, index=False, encoding='gb18030')
    entry('Chinese GB18030 date notation', path, no_cop=True)
    path = destination / 'synthetic-normalized-collision.csv'
    pd.DataFrame({'time': pd.date_range('2025-05-01', periods=3, freq='h'),
                  'CH-907 LWT ℃': [6., 6., 6.], 'CH-907 LWT °C': [7., 7., 7.]}).to_csv(path, index=False)
    entry('Unicode header collision', path, no_cop=True)
    path = destination / 'synthetic-legacy.xls'
    import xlwt
    book = xlwt.Workbook(); sheet = book.add_sheet('SYNTHETIC TEST DATA')
    for col, name in enumerate(english.columns): sheet.write(0, col, name)
    for row, values in enumerate(english.itertuples(index=False, name=None), 1):
        for col, value in enumerate(values): sheet.write(row, col, value)
    book.save(str(path)); entry('Legacy XLS', path, expect_cop=True)
    path = destination / 'synthetic-invalid.xlsx'; path.write_text('SYNTHETIC INVALID WORKBOOK', encoding='utf-8')
    entry('Invalid workbook', path, expect_error=True)
    path = destination / 'synthetic-malformed.csv'; path.write_text('time,power\n2025-01-01,2\n2025-01-02,3,unexpected\n', encoding='utf-8')
    entry('Malformed CSV', path, expect_error=True)
    path = destination / 'synthetic-combined.csv'
    combined = pd.DataFrame({'timestamp': pd.date_range('2025-06-01', periods=96, freq='15min'),
        'AHP-901 往水温度 (°C)': [6.0] * 96, 'AHP-901 還水温度 (°C)': [11.0] * 96,
        'AHP-901 流量 (m3/h)': [18.] * 96, 'AHP-901 消費電力 (kW)': [30.] * 96})
    combined.to_csv(path, index=False)
    reviews = [{'column': name, 'label': role, 'equipment': 'AHP-901', 'unit': unit} for name, role, unit in zip(combined.columns[1:],
        ('heat_source_supply_temp', 'heat_source_return_temp', 'heat_source_flow', 'heat_source_power'), ('°C', '°C', 'm3/h', 'kW'))]
    entry('Combined BEMS drawing/model references', path, expect_cop=True, reviews=reviews)
    return entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps({'bems': create(args.destination)}, ensure_ascii=False, indent=2))
