def launch_multi_panel(self):
        """Launch multi-panel chart with subcharts"""
        if self.raw_data is None:
            messagebox.showerror("Error", "No data available. Please check the CSV file.")
            return
            
        timeframe = self.timeframe_var.get()
        
        # Get data for timeframe
        data = self.get_timeframe_data(timeframe)
        if data is None or len(data) == 0:
            messagebox.showerror("Error", "No data available for selected timeframe.")
            return
            
        self.log_message(f"Creating multi-panel chart for: {timeframe}")
        
        # Create main chart
        chart = self.create_styled_chart(f"Multi-Panel Analysis - {timeframe}")
        chart.set(data)
        
        # Add moving averages to main chart
        if len(data) >= 20:
            sma20 = chart.create_line('SMA 20', color='#FF6B6B', width=2)
            sma20_data = self.calculate_sma(data, 20)
            sma20.set(sma20_data)
            
        if len(data) >= 50:
            sma50 = chart.create_line('SMA 50', color='#4ECDC4', width=2)
            sma50_data = self.calculate_sma(data, 50)
            sma50.set(sma50_data)
        
        # Add VWAP if selected
        if self.indicator_vars['VWAP'].get():
            vwap = chart.create_line('VWAP', color='#FFA726', width=2)
            vwap_data = self.calculate_vwap(data)
            vwap.set(vwap_data)
        
        # Create RSI subchart
        if len(data) > 14:
            rsi_chart = chart.create_subchart(width=1, height=0.3, sync_crosshairs=True)
            rsi_chart.layout(background_color='#131722', text_color='#d1d4dc')
            rsi_chart.price_scale(scale_margin_top=0.1, scale_margin_bottom=0.1)
            
            rsi_line = rsi_chart.create_line('RSI', color='#9C27B0', width=2)
            rsi_data = self.calculate_rsi(data)
            rsi_line.set(rsi_data)
            
            # Add RSI levels
            rsi_chart.horizontal_line(70, color='#FF4444', width=1, style='dashed', text='Overbought')
            rsi_chart.horizontal_line(30, color='#44FF44', width=1, style='dashed', text='Oversold')
            rsi_chart.horizontal_line(50, color='#FFFF44', width=1, style='dotted', text='Midline')
        
        # Create MACD subchart
        if len(data) > 26:
            macd_chart = chart.create_subchart(width=1, height=0.3, sync_crosshairs=True)
            macd_chart.layout(background_color='#131722', text_color='#d1d4dc')
            
            macd_data = self.calculate_macd(data)
            
            macd_line = macd_chart.create_line('MACD', color='#2196F3', width=2)
            macd_line.set(macd_data['macd'])
            
            signal_line = macd_chart.create_line('Signal', color='#FF9800', width=2)
            signal_line.set(macd_data['signal'])
            
            histogram = macd_chart.create_histogram('Histogram', color='#9E9E9E')
            histogram.set(macd_data['histogram'])
            
            # Zero line
            macd_chart.horizontal_line(0, color='#666666', width=1, style='dotted')
        
        self.log_message(f"Multi-panel chart launched for {timeframe}")
        chart.show(block=False)
        
    def start_simulation(self):
        """Start a simulation mode showing data progression"""
        if self.raw_data is None:
            messagebox.showerror("Error", "No data available. Please check the CSV file.")
            return
            
        if self.is_running:
            self.is_running = False
            self.log_message("Stopping simulation...")
            return
            
        self.is_running = True
        self.log_message("Starting simulation mode...")
        threading.Thread(target=self.run_simulation, daemon=True).start()
        
    def run_simulation(self):
        """Run simulation showing data progression over time"""
        try:
            # Use last 1000 points for simulation
            sim_data = self.raw_data.tail(1000).copy()
            
            # Create chart for simulation
            chart = self.create_styled_chart("Live Simulation Mode")
            
            # Start with first 100 points
            initial_data = sim_data.head(100)
            chart.set(initial_data[['time', 'open', 'high', 'low', 'close', 'volume']])
            
            # Add SMA line
            sma_line = chart.create_line('SMA 20', color='#FF6B6B', width=2)
            
            chart.show(block=False)
            
            self.log_message("Simulation started - showing data progression...")
            
            # Add remaining data points one by one
            for i in range(100, len(sim_data)):
                if not self.is_running:
                    break
                    
                new_row = sim_data.iloc[i]
                chart.update(new_row)
                
                # Update SMA every 20 points
                if i % 20 == 0:
                    current_data = sim_data.head(i + 1)
                    sma_data = self.calculate_sma(current_data, 20)
                    if len(sma_data) > 0:
                        sma_line.set(sma_data)
                
                # Add markers for significant moves
                if i > 0:
                    prev_close = sim_data.iloc[i-1]['close']
                    current_close = new_row['close']
                    change_pct = ((current_close - prev_close) / prev_close) * 100
                    
                    if abs(change_pct) > 1:  # 1% movement
                        color = '#00FF00' if change_pct > 0 else '#FF0000'
                        position = 'above' if change_pct > 0 else 'below'
                        chart.marker(
                            time=datetime.fromtimestamp(new_row['time']),
                            position=position,
                            shape='circle',
                            color=color,
                            text=f'{change_pct:+.1f}%'
                        )
                
                sleep(0.1)  # 100ms delay between updates
                
            self.log_message("Simulation completed!")
            self.is_running = False
            
        except Exception as e:
            self.log_message(f"Simulation error: {str(e)}")
            self.is_running = False#!/usr/bin/env python3
"""
Lightweight Charts Python Dashboard
A comprehensive financial dashboard using the lightweight-charts-python library
Data Source: Custom CSV file with 5-minute OHLC data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from time import sleep
from lightweight_charts import Chart
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import queue
import os

class FinancialDashboard:
    def __init__(self, csv_file='Latest file 1.csv'):
        self.csv_file = csv_file
        self.charts = {}
        self.data_queue = queue.Queue()
        self.is_running = False
        self.raw_data = None
        
        # Available timeframes based on 5-minute data
        self.timeframes = {
            '1 Day': 288,      # 288 * 5min = 1 day
            '3 Days': 864,     # 3 days
            '1 Week': 2016,    # 7 days
            '2 Weeks': 4032,   # 14 days
            '1 Month': 8640,   # 30 days
            '3 Months': 25920, # 90 days
            '6 Months': 51840, # 180 days
            '1 Year': 103680,  # 360 days
            'All Data': -1     # All available data
        }
        self.current_timeframe = '1 Month'
        
        # Technical indicators
        self.indicators = {
            'SMA_20': True,
            'SMA_50': True,
            'EMA_12': False,
            'EMA_26': False,
            'RSI': False,
            'MACD': False,
            'Bollinger_Bands': False,
            'VWAP': False
        }
        
        # Load and prepare data
        self.load_data()
    def load_data(self):
        """Load and prepare data from CSV file"""
        try:
            if not os.path.exists(self.csv_file):
                messagebox.showerror("Error", f"CSV file '{self.csv_file}' not found!")
                return False
                
            self.log_message(f"Loading data from {self.csv_file}...")
            
            # Read CSV file
            self.raw_data = pd.read_csv(self.csv_file)
            
            # Convert date and time to datetime
            self.raw_data['datetime'] = pd.to_datetime(
                self.raw_data['date'] + ' ' + self.raw_data['time'], 
                format='%d-%m-%Y %H:%M:%S'
            )
            
            # Convert to Unix timestamp for lightweight charts
            self.raw_data['time'] = self.raw_data['datetime'].astype('int64') // 10**9
            
            # Rename columns to match lightweight charts format
            column_mapping = {
                'Volume': 'volume',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close'
            }
            self.raw_data.rename(columns=column_mapping, inplace=True)
            
            # Sort by datetime
            self.raw_data.sort_values('datetime', inplace=True)
            self.raw_data.reset_index(drop=True, inplace=True)
            
            # Get data info
            start_date = self.raw_data['datetime'].min()
            end_date = self.raw_data['datetime'].max()
            total_points = len(self.raw_data)
            
            self.log_message(f"Data loaded successfully!")
            self.log_message(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            self.log_message(f"Total data points: {total_points:,}")
            self.log_message(f"Frequency: 5-minute intervals")
            
            return True
            
        except Exception as e:
            self.log_message(f"Error loading data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            return False
        
    def setup_gui(self):
        """Setup the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("Lightweight Charts Financial Dashboard - Custom Data")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Dashboard Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Data source info
        ttk.Label(control_frame, text="Data Source:").grid(row=0, column=0, padx=5, sticky='w')
        ttk.Label(control_frame, text=self.csv_file, font=('Arial', 9, 'bold')).grid(row=0, column=1, padx=5, sticky='w')
        
        # Timeframe selection
        ttk.Label(control_frame, text="Timeframe:").grid(row=0, column=2, padx=5, sticky='w')
        self.timeframe_var = tk.StringVar(value=self.current_timeframe)
        timeframe_combo = ttk.Combobox(control_frame, textvariable=self.timeframe_var,
                                      values=list(self.timeframes.keys()), width=12)
        timeframe_combo.grid(row=0, column=3, padx=5)
        timeframe_combo.bind('<<ComboboxSelected>>', self.on_timeframe_change)
        
        # Buttons
        ttk.Button(control_frame, text="Launch Chart", 
                  command=self.launch_chart).grid(row=0, column=4, padx=10)
        ttk.Button(control_frame, text="Multi-Panel View", 
                  command=self.launch_multi_panel).grid(row=0, column=5, padx=5)
        ttk.Button(control_frame, text="Simulation Mode", 
                  command=self.start_simulation).grid(row=0, column=6, padx=5)
        
        # Data range info
        data_info_frame = ttk.LabelFrame(main_frame, text="Data Information", padding=10)
        data_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        if self.raw_data is not None:
            start_date = self.raw_data['datetime'].min().strftime('%Y-%m-%d %H:%M')
            end_date = self.raw_data['datetime'].max().strftime('%Y-%m-%d %H:%M')
            total_points = len(self.raw_data)
            
            ttk.Label(data_info_frame, text=f"Period: {start_date} to {end_date}").pack(anchor='w')
            ttk.Label(data_info_frame, text=f"Total Points: {total_points:,} | Frequency: 5-minute intervals").pack(anchor='w')
        
        # Indicators panel
        indicators_frame = ttk.LabelFrame(main_frame, text="Technical Indicators", padding=10)
        indicators_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.indicator_vars = {}
        row = 0
        col = 0
        for indicator, enabled in self.indicators.items():
            var = tk.BooleanVar(value=enabled)
            self.indicator_vars[indicator] = var
            ttk.Checkbutton(indicators_frame, text=indicator.replace('_', ' '), 
                           variable=var).grid(row=row, column=col, padx=10, sticky='w')
            col += 1
            if col > 3:  # 4 indicators per row
                col = 0
                row += 1
        
        # Status panel
        status_frame = ttk.LabelFrame(main_frame, text="Dashboard Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_text = tk.Text(status_frame, height=15, width=80)
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_message("Financial Dashboard initialized with custom data source")
        if self.raw_data is not None:
            self.log_message("Data loaded successfully - Ready to create charts!")
        
    def log_message(self, message):
        """Add message to status log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
    def get_timeframe_data(self, timeframe_key):
        """Get data for specified timeframe"""
        if self.raw_data is None:
            return None
            
        periods = self.timeframes[timeframe_key]
        
        if periods == -1:  # All data
            data = self.raw_data.copy()
        else:
            # Get the last N periods
            data = self.raw_data.tail(periods).copy()
            
        # Select required columns for lightweight charts
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        return data[required_cols].copy()
            
    def calculate_sma(self, data, period):
        """Calculate Simple Moving Average"""
        result = data.copy()
        result['value'] = result['close'].rolling(window=period).mean()
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': pd.DataFrame({'time': data['time'], 'value': upper_band}).dropna(),
            'middle': pd.DataFrame({'time': data['time'], 'value': sma}).dropna(),
            'lower': pd.DataFrame({'time': data['time'], 'value': lower_band}).dropna()
        }
        
    def calculate_vwap(self, data):
        """Calculate Volume Weighted Average Price"""
        # Handle zero volume data
        volume = data['volume'].replace(0, 1)  # Replace 0 with 1 to avoid division by zero
        
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        cumulative_volume = volume.cumsum()
        cumulative_tp_volume = (typical_price * volume).cumsum()
        
        vwap = cumulative_tp_volume / cumulative_volume
        
        return pd.DataFrame({'time': data['time'], 'value': vwap}).dropna()
        
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        result = data.copy()
        result['value'] = result['close'].ewm(span=period).mean()
        return result[['time', 'value']].dropna()
        
    def calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        result = data.copy()
        result['value'] = rsi
        return result[['time', 'value']].dropna()
        
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        result = data.copy()
        result['macd'] = macd_line
        result['signal'] = signal_line
        result['histogram'] = histogram
        
        return {
            'macd': result[['time', 'macd']].rename(columns={'macd': 'value'}).dropna(),
            'signal': result[['time', 'signal']].rename(columns={'signal': 'value'}).dropna(),
            'histogram': result[['time', 'histogram']].rename(columns={'histogram': 'value'}).dropna()
        }
        
    def create_styled_chart(self, title="Financial Chart", toolbox=True):
        """Create a styled chart with professional appearance"""
        chart = Chart(toolbox=toolbox)
        
        # Professional dark theme
        chart.layout(
            background_color='#131722',
            text_color='#d1d4dc',
            font_size=12,
            font_family='Trebuchet MS'
        )
        
        # Candlestick styling
        chart.candle_style(
            up_color='#26a69a',
            down_color='#ef5350',
            border_up_color='#26a69a',
            border_down_color='#ef5350',
            wick_up_color='#26a69a',
            wick_down_color='#ef5350'
        )
        
        # Volume styling
        chart.volume_config(
            up_color='rgba(38, 166, 154, 0.7)',
            down_color='rgba(239, 83, 80, 0.7)'
        )
        
        # Crosshair
        chart.crosshair(
            mode='normal',
            vert_color='#758696',
            vert_style='dotted',
            horz_color='#758696',
            horz_style='dotted'
        )
        
        # Watermark
        chart.watermark(title, color='rgba(180, 180, 240, 0.3)')
        
        # Legend
        chart.legend(visible=True, font_size=11)
        
        return chart
        
    def launch_chart(self):
        """Launch the main chart window"""
        if self.raw_data is None:
            messagebox.showerror("Error", "No data available. Please check the CSV file.")
            return
            
        timeframe = self.timeframe_var.get()
        
        # Get data for timeframe
        data = self.get_timeframe_data(timeframe)
        if data is None or len(data) == 0:
            messagebox.showerror("Error", "No data available for selected timeframe.")
            return
            
        self.log_message(f"Creating chart for timeframe: {timeframe}")
        self.log_message(f"Data points: {len(data):,}")
        
        # Create chart
        chart = self.create_styled_chart(f"Financial Data - {timeframe}")
        chart.set(data)
        
        # Add technical indicators
        indicators_added = []
        
        if self.indicator_vars['SMA_20'].get() and len(data) >= 20:
            sma20 = chart.create_line('SMA 20', color='#FF6B6B', width=2)
            sma20_data = self.calculate_sma(data, 20)
            sma20.set(sma20_data)
            indicators_added.append('SMA 20')
            
        if self.indicator_vars['SMA_50'].get() and len(data) >= 50:
            sma50 = chart.create_line('SMA 50', color='#4ECDC4', width=2)
            sma50_data = self.calculate_sma(data, 50)
            sma50.set(sma50_data)
            indicators_added.append('SMA 50')
            
        if self.indicator_vars['EMA_12'].get() and len(data) >= 12:
            ema12 = chart.create_line('EMA 12', color='#45B7D1', width=2)
            ema12_data = self.calculate_ema(data, 12)
            ema12.set(ema12_data)
            indicators_added.append('EMA 12')
            
        if self.indicator_vars['EMA_26'].get() and len(data) >= 26:
            ema26 = chart.create_line('EMA 26', color='#F7DC6F', width=2)
            ema26_data = self.calculate_ema(data, 26)
            ema26.set(ema26_data)
            indicators_added.append('EMA 26')
            
        if self.indicator_vars['Bollinger_Bands'].get() and len(data) >= 20:
            bb_data = self.calculate_bollinger_bands(data, 20)
            
            bb_upper = chart.create_line('BB Upper', color='#9C27B0', width=1, style='dashed')
            bb_upper.set(bb_data['upper'])
            
            bb_middle = chart.create_line('BB Middle', color='#9C27B0', width=1)
            bb_middle.set(bb_data['middle'])
            
            bb_lower = chart.create_line('BB Lower', color='#9C27B0', width=1, style='dashed')
            bb_lower.set(bb_data['lower'])
            
            indicators_added.append('Bollinger Bands')
            
        if self.indicator_vars['VWAP'].get():
            vwap = chart.create_line('VWAP', color='#FFA726', width=2)
            vwap_data = self.calculate_vwap(data)
            vwap.set(vwap_data)
            indicators_added.append('VWAP')
        
        # Add support/resistance levels based on recent highs/lows
        recent_data = data.tail(100)  # Last 100 periods
        resistance_level = recent_data['high'].max()
        support_level = recent_data['low'].min()
        
        chart.horizontal_line(
            resistance_level, 
            color='#FF4444', 
            width=2, 
            style='dashed',
            text=f'Resistance: {resistance_level:.2f}'
        )
        chart.horizontal_line(
            support_level, 
            color='#44FF44', 
            width=2, 
            style='dashed',
            text=f'Support: {support_level:.2f}'
        )
        
        self.log_message(f"Chart launched for {timeframe}")
        if indicators_added:
            self.log_message(f"Indicators added: {', '.join(indicators_added)}")
            
        # Store chart reference
        self.charts[timeframe] = chart
        
        # Show chart
        chart.show(block=False)
        
    def launch_multi_panel(self):
        """Launch multi-panel chart with subcharts"""
        symbol = self.symbol_var.get()
        timeframe = self.timeframe_var.get()
        
        # Fetch data
        data = self.fetch_data(symbol, timeframe)
        if data is None:
            return
            
        # Create main chart
        chart = self.create_styled_chart(f"{symbol} Multi-Panel Analysis")
        chart.set(data)
        
        # Add moving averages to main chart
        sma20 = chart.create_line('SMA 20', color='#FF6B6B', width=2)
        sma20_data = self.calculate_sma(data, 20)
        sma20.set(sma20_data)
        
        # Create RSI subchart
        if len(data) > 14:  # Ensure we have enough data for RSI
            rsi_chart = chart.create_subchart(width=1, height=0.3, sync_crosshairs=True)
            rsi_chart.layout(background_color='#131722', text_color='#d1d4dc')
            rsi_chart.price_scale(scale_margin_top=0.1, scale_margin_bottom=0.1)
            
            rsi_line = rsi_chart.create_line('RSI', color='#9C27B0', width=2)
            rsi_data = self.calculate_rsi(data)
            rsi_line.set(rsi_data)
            
            # Add RSI levels
            rsi_chart.horizontal_line(70, color='#FF4444', width=1, style='dashed', text='Overbought')
            rsi_chart.horizontal_line(30, color='#44FF44', width=1, style='dashed', text='Oversold')
            rsi_chart.horizontal_line(50, color='#FFFF44', width=1, style='dotted', text='Midline')
        
        # Create MACD subchart
        if len(data) > 26:  # Ensure we have enough data for MACD
            macd_chart = chart.create_subchart(width=1, height=0.3, sync_crosshairs=True)
            macd_chart.layout(background_color='#131722', text_color='#d1d4dc')
            
            macd_data = self.calculate_macd(data)
            
            macd_line = macd_chart.create_line('MACD', color='#2196F3', width=2)
            macd_line.set(macd_data['macd'])
            
            signal_line = macd_chart.create_line('Signal', color='#FF9800', width=2)
            signal_line.set(macd_data['signal'])
            
            histogram = macd_chart.create_histogram('Histogram', color='#9E9E9E')
            histogram.set(macd_data['histogram'])
            
            # Zero line
            macd_chart.horizontal_line(0, color='#666666', width=1, style='dotted')
        
        self.log_message(f"Multi-panel chart launched for {symbol}")
        chart.show(block=False)
        
    def on_timeframe_change(self, event):
        """Handle timeframe change"""
        self.current_timeframe = self.timeframe_var.get()
        periods = self.timeframes[self.current_timeframe]
        period_text = f"{periods:,} periods" if periods != -1 else "all data"
        self.log_message(f"Timeframe changed to: {self.current_timeframe} ({period_text})")
        
    def run(self):
        """Run the dashboard"""
        if self.raw_data is None:
            self.log_message("Error: Could not load data. Please check the CSV file path.")
            messagebox.showerror("Error", "Could not load data. Please check the CSV file.")
        else:
            self.log_message("Dashboard ready - Select timeframe and launch charts!")
        self.root.mainloop()

def demo_csv_chart(csv_file='Latest file 1.csv'):
    """Demo chart using CSV data"""
    print("Creating chart from CSV data...")
    
    try:
        # Load CSV data
        if not os.path.exists(csv_file):
            print(f"Error: CSV file '{csv_file}' not found!")
            return
            
        data = pd.read_csv(csv_file)
        
        # Convert date and time to datetime
        data['datetime'] = pd.to_datetime(
            data['date'] + ' ' + data['time'], 
            format='%d-%m-%Y %H:%M:%S'
        )
        
        # Convert to Unix timestamp
        data['time'] = data['datetime'].astype('int64') // 10**9
        
        # Rename columns and take last 1000 points for demo
        data.rename(columns={'Volume': 'volume'}, inplace=True)
        demo_data = data.tail(1000)[['time', 'open', 'high', 'low', 'close', 'volume']]
        
        # Create and configure chart
        chart = Chart(toolbox=True)
        
        # Professional styling
        chart.layout(
            background_color='#131722',
            text_color='#d1d4dc',
            font_size=12,
            font_family='Trebuchet MS'
        )
        
        chart.candle_style(
            up_color='#26a69a',
            down_color='#ef5350',
            border_up_color='#26a69a',
            border_down_color='#ef5350',
            wick_up_color='#26a69a',
            wick_down_color='#ef5350'
        )
        
        chart.volume_config(
            up_color='rgba(38, 166, 154, 0.7)',
            down_color='rgba(239, 83, 80, 0.7)'
        )
        
        # Set data
        chart.set(demo_data)
        
        # Add SMA
        sma_data = pd.DataFrame({
            'time': demo_data['time'],
            'value': demo_data['close'].rolling(window=20).mean()
        }).dropna()
        
        sma_line = chart.create_line('SMA 20', color='#FF6B6B', width=2)
        sma_line.set(sma_data)
        
        # Add styling
        chart.watermark('CSV Data Demo', color='rgba(180, 180, 240, 0.3)')
        chart.legend(visible=True)
        chart.crosshair(mode='normal', vert_color='#758696', horz_color='#758696')
        
        print(f"Demo chart created with {len(demo_data)} data points!")
        print("Close the chart window to continue...")
        chart.show(block=True)
        
    except Exception as e:
        print(f"Error creating demo chart: {str(e)}")

if __name__ == "__main__":
    print("Lightweight Charts Python Dashboard - CSV Data Source")
    print("=====================================================")
    print()
    print("This dashboard uses your custom CSV file as the data source:")
    print("- High-frequency 5-minute OHLC data")
    print("- Multiple timeframe views")
    print("- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, VWAP)")
    print("- Multi-panel charts with subcharts")
    print("- Simulation mode for data progression")
    print("- Professional TradingView-style interface")
    print()
    
    csv_file = 'Latest file 1.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        print("Please ensure the CSV file is in the same directory as this script.")
        choice = input("\nWould you like to see a demo with sample data instead? (y/n): ")
        if choice.lower() == 'y':
            # Create sample data for demo
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='5T')
            np.random.seed(42)
            
            data = []
            price = 15000  # Starting price similar to your data
            
            for i, date in enumerate(dates):
                if i >= 1000:  # Limit to 1000 points for demo
                    break
                    
                change = np.random.normal(0, 10)
                price += change
                
                open_price = price + np.random.normal(0, 5)
                high_price = max(open_price, price) + abs(np.random.normal(0, 10))
                low_price = min(open_price, price) - abs(np.random.normal(0, 10))
                close_price = price + np.random.normal(0, 5)
                volume = np.random.randint(0, 1000)
                
                data.append({
                    'time': int(date.timestamp()),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': volume
                })
            
            demo_df = pd.DataFrame(data)
            chart = Chart(toolbox=True)
            chart.layout(background_color='#131722', text_color='#d1d4dc')
            chart.candle_style(up_color='#26a69a', down_color='#ef5350')
            chart.set(demo_df)
            chart.watermark('Sample Data Demo', color='rgba(180, 180, 240, 0.3)')
            chart.legend(visible=True)
            chart.show(block=True)
    else:
        choice = input("\nChoose an option:\n1. Full Dashboard (GUI)\n2. Quick Chart Demo\nEnter choice (1 or 2): ")
        
        if choice == "2":
            demo_csv_chart(csv_file)
        else:
            dashboard = FinancialDashboard(csv_file)
            dashboard.run()
