import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math 

# ================================
#   ПЕРЕВОД ПИКСЕЛЕЙ → САНТИМЕТРЫ
# ================================
def pixel_to_cm(error_px, height_cm=6.0, tilt_deg=70.0, fx_px=3214.0):
    """
    Конвертация ошибки из пикселей в сантиметры.
    height_cm — расстояние камеры до пола
    tilt_deg — угол наклона камеры вниз
    fx_px — фокальное расстояние IMX219 (≈ 3214 px)
    """
    tilt_rad = math.radians(tilt_deg)
    scale_cm_per_px = height_cm / (fx_px * math.cos(tilt_rad))
    return error_px * scale_cm_per_px


def visualize_telemetry(csv_path, output_dir="./plots", max_frames=1400, window_size=50):
    """
    Визуализация телеметрии линейного следования.
    Создаёт отдельные графики для каждой метрики.
    
    Args:
        csv_path: путь к CSV файлу с телеметрией
        output_dir: директория для сохранения графиков
        max_frames: максимальное количество кадров для анализа (None = все)
        window_size: размер окна для скользящего среднего
    """
    # Создаём директорию для графиков
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Загрузка данных с ограничением
    if max_frames is not None:
        df = pd.read_csv(csv_path, nrows=max_frames)
        print(f"Loaded {len(df)} frames (limited to {max_frames}) from {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} frames from {csv_path}")
    
    # Базовый стиль
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        'raw': '#e74c3c',
        'filtered': '#3498db',
        'mean': '#2ecc71',
        'steady': '#9b59b6'
    }
    
    # === 1. ОШИБКА: raw vs filtered === #
    fig, ax = plt.subplots(figsize=(14, 6))
        # === Добавляем ошибку в сантиметрах ===
    df['error_cm_raw'] = df['error_raw'].apply(lambda px: pixel_to_cm(px))
    df['error_cm_FIL'] = df['error_filtered'].apply(lambda px: pixel_to_cm(px))
    
    ax.plot(df['frame'], df['error_cm_raw'], 
            label='Raw Error', color=colors['raw'], alpha=0.5, linewidth=1)
    ax.plot(df['frame'], df['error_cm_FIL'], 
            label='Filtered Error', color=colors['filtered'], linewidth=2)
    
    # Скользящие средние значения
    rolling_mean_raw = df['error_cm_raw'].rolling(window=window_size, center=True).mean()
    rolling_mean_filtered = df['error_cm_FIL'].rolling(window=window_size, center=True).mean()
    
    ax.plot(df['frame'], rolling_mean_raw, color=colors['mean'], linestyle='--', 
            label=f'Rolling Mean Raw (window={window_size})', linewidth=2, alpha=0.8)
    ax.plot(df['frame'], rolling_mean_filtered, color=colors['steady'], linestyle='--', 
            label=f'Rolling Mean Filtered (window={window_size})', linewidth=2, alpha=0.8)
    
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Кадр', fontsize=12)
    ax.set_ylabel('Ошибка (см)', fontsize=12)
    ax.set_title("Ошибка следования за линией", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: error_comparison.png")
    
    # === 2. УСТАНОВИВШАЯСЯ ОШИБКА (steady state) === #
    # Вычисляем экспоненциально сглаженную ошибку (EMA)
    steady_error = df['error_filtered'].ewm(alpha=0.05).mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['frame'], df['error_filtered'], 
            label='Filtered Error', color=colors['filtered'], alpha=0.6, linewidth=1.5)
    ax.plot(df['frame'], steady_error, 
            label='Steady State Error (EMA α=0.05)', 
            color=colors['steady'], linewidth=2.5)
    
    # Финальное установившееся значение
    final_steady = steady_error.iloc[-1]
    ax.axhline(final_steady, color=colors['steady'], linestyle=':', 
               label=f'Final Steady ({final_steady:.2f})', linewidth=2)
    
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Steady State Error Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'steady_state_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: steady_state_error.png")
    
    # === 3. УГОЛ ЛИНИИ === #
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['frame'], df['angle_deg'], 
            label='Line Angle', color='#e67e22', linewidth=1.5, alpha=0.6)
    
    # Скользящее среднее для угла
    rolling_mean_angle = df['angle_deg'].rolling(window=window_size, center=True).mean()
    ax.plot(df['frame'], rolling_mean_angle, color='#d35400', linestyle='--',
            label=f'Rolling Mean (window={window_size})', linewidth=2)
    
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_title('Line Angle Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'line_angle.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: line_angle.png")
    
    # === 4. СКОРОСТЬ (V) === #
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['frame'], df['v'], 
            label='Velocity', color='#16a085', linewidth=1.5, alpha=0.6)
    
    # Скользящее среднее для скорости
    rolling_mean_v = df['v'].rolling(window=window_size, center=True).mean()
    ax.plot(df['frame'], rolling_mean_v, color='#0e6655', linestyle='--',
            label=f'Rolling Mean (window={window_size})', linewidth=2)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Velocity (PWM)', fontsize=12)
    ax.set_title('Robot Velocity', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'velocity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: velocity.png")
    
    # === 5. БУСТ СКОРОСТИ === #
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['frame'], df['straight_boost'], 
            label='Straight Boost', color='#f39c12', linewidth=1.5, alpha=0.6)
    ax.fill_between(df['frame'], 0, df['straight_boost'], 
                     color='#f39c12', alpha=0.2)
    
    # Скользящее среднее для буста
    rolling_mean_boost = df['straight_boost'].rolling(window=window_size, center=True).mean()
    ax.plot(df['frame'], rolling_mean_boost, color='#e67e22', linestyle='--',
            label=f'Rolling Mean (window={window_size})', linewidth=2)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Boost (PWM)', fontsize=12)
    ax.set_title('Speed Boost on Straight Sections', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_boost.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: speed_boost.png")
    
    # === 6. КОМАНДЫ МОТОРАМ === #
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['frame'], df['left_cmd'], 
            label='Left Motor', color='#c0392b', linewidth=1.5, alpha=0.5)
    ax.plot(df['frame'], df['right_cmd'], 
            label='Right Motor', color='#2980b9', linewidth=1.5, alpha=0.5)
    
    # Скользящие средние для команд моторам
    rolling_mean_left = df['left_cmd'].rolling(window=window_size, center=True).mean()
    rolling_mean_right = df['right_cmd'].rolling(window=window_size, center=True).mean()
    
    ax.plot(df['frame'], rolling_mean_left, color='#a93226', linestyle='--',
            label=f'Left Rolling Mean (window={window_size})', linewidth=2)
    ax.plot(df['frame'], rolling_mean_right, color='#1f618d', linestyle='--',
            label=f'Right Rolling Mean (window={window_size})', linewidth=2)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Motor Command (PWM)', fontsize=12)
    ax.set_title('Motor Commands', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'motor_commands.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: motor_commands.png")
    
    # === 7. FPS === #
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['frame'], df['fps'], 
            label='FPS', color='#8e44ad', linewidth=1.5, alpha=0.6)
    
    # Скользящее среднее для FPS
    rolling_mean_fps = df['fps'].rolling(window=window_size, center=True).mean()
    ax.plot(df['frame'], rolling_mean_fps, color='#6c3483', linestyle='--',
            label=f'Rolling Mean (window={window_size})', linewidth=2)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Frames Per Second', fontsize=12)
    ax.set_title('Processing Speed (FPS)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fps.png")
    
    # === 8. НАПРАВЛЕНИЕ (гистограмма) === #
    direction_counts = df['direction'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(direction_counts.index, direction_counts.values, 
                   color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Direction', fontsize=12)
    ax.set_ylabel('Frame Count', fontsize=12)
    ax.set_title('Direction Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'direction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: direction_distribution.png")
    
    # === 9. СВОДНАЯ СТАТИСТИКА === #
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Телеметрия робота', fontsize=16, fontweight='bold', y=0.995)
    
    # 9.1 Ошибка
    axes[0, 0].plot(df['frame'], df['error_raw'], 'r-', alpha=0.3, linewidth=1, label='Raw')
    axes[0, 0].plot(df['frame'], df['error_filtered'], 'b-', linewidth=2, label='Filtered')
    axes[0, 0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0, 0].set_title('Error', fontweight='bold')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 9.2 Моторы
    axes[0, 1].plot(df['frame'], df['left_cmd'], 'r-', linewidth=1.5, label='Left')
    axes[0, 1].plot(df['frame'], df['right_cmd'], 'b-', linewidth=1.5, label='Right')
    axes[0, 1].set_title('Motor Commands', fontweight='bold')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('PWM')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 9.3 Скорость + Угол
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(df['frame'], df['v'], 'g-', linewidth=1.5, label='Velocity')
    line2 = ax2.plot(df['frame'], df['angle_deg'], 'orange', linewidth=1.5, label='Angle')
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Velocity (PWM)', color='g')
    ax2.set_ylabel('Angle (deg)', color='orange')
    ax1.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.set_title('Velocity & Angle', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 9.4 Статистика (таблица)
    axes[1, 1].axis('off')
    
    stats_data = [
        ['Metric', 'Mean', 'Std', 'Min', 'Max'],
        ['Error (filtered)', f'{df["error_filtered"].mean():.2f}', 
         f'{df["error_filtered"].std():.2f}', 
         f'{df["error_filtered"].min():.2f}', 
         f'{df["error_filtered"].max():.2f}'],
        ['Angle (deg)', f'{df["angle_deg"].mean():.1f}', 
         f'{df["angle_deg"].std():.1f}', 
         f'{df["angle_deg"].min():.1f}', 
         f'{df["angle_deg"].max():.1f}'],
        ['Velocity', f'{df["v"].mean():.1f}', 
         f'{df["v"].std():.1f}', 
         f'{df["v"].min():.1f}', 
         f'{df["v"].max():.1f}'],
        ['FPS', f'{df["fps"].mean():.1f}', 
         f'{df["fps"].std():.1f}', 
         f'{df["fps"].min():.1f}', 
         f'{df["fps"].max():.1f}'],
    ]
    
    table = axes[1, 1].table(cellText=stats_data, cellLoc='center', loc='center',
                              colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Заголовок таблицы
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Statistics Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: summary_dashboard.png")
    
    # === ИТОГОВЫЙ ОТЧЁТ === #
    print("\n" + "="*60)
    print("TELEMETRY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total frames: {len(df)}")
    print(f"Average FPS: {df['fps'].mean():.1f} ± {df['fps'].std():.1f}")
    print(f"\nError (filtered):")
    print(f"  Mean: {df['error_filtered'].mean():.3f}")
    print(f"  Std:  {df['error_filtered'].std():.3f}")
    print(f"  Steady state (final): {steady_error.iloc[-1]:.3f}")
    print(f"\nDirection distribution:")
    for direction, count in direction_counts.items():
        print(f"  {direction}: {count} frames ({count/len(df)*100:.1f}%)")
    print(f"\nPlots saved to: {output_dir.absolute()}")
    print("="*60)

if __name__ == "__main__":
    # Пример использования
    csv_path = "/home/angelika/Desktop/Seoul/Vision-Based-Line-Following-Car/data/for_analysis/linetracer_thresh_20251119_153730_labels.txt" 
    visualize_telemetry(csv_path, output_dir="./telemetry_plots", max_frames=560, window_size=20)