import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math 

# ============================
#   ПИКСЕЛИ → САНТИМЕТРЫ
# ============================
def pixel_to_cm(error_px, height_cm=6.0, tilt_deg=70.0, fx_px=3214.0):
    """
    Перевод ошибки из пикселей в см.
    height_cm — высота камеры
    tilt_deg — наклон вниз
    fx_px — фокальное расстояние IMX219
    """
    tilt_rad = math.radians(tilt_deg)
    scale = height_cm / (fx_px * math.cos(tilt_rad))
    return error_px * scale


def visualize_telemetry(csv_path, output_dir="./plots", max_frames=1400, window_size=50):
    """
    Основная функция визуализации.
    Строит графики ошибок, угла, скорости, команд моторам, FPS и т.п.
    """
    # Создание директории
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Загрузка данных
    if max_frames is not None:
        df = pd.read_csv(csv_path, nrows=max_frames)
        print(f"Загружено {len(df)} кадров (лимит {max_frames})")
    else:
        df = pd.read_csv(csv_path)
        print(f"Загружено {len(df)} кадров")

    # Темы и цвета
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        'raw': '#e74c3c',
        'filtered': '#3498db',
        'mean': '#2ecc71',
        'steady': '#9b59b6'
    }

    # ===================================
    #   1. Ошибка (в см): raw / filtered
    # ===================================
    df['E_raw_cm'] = df['error_raw'].apply(pixel_to_cm)
    df['E_fil_cm'] = df['error_filtered'].apply(pixel_to_cm)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df['frame'], df['E_raw_cm'], label='E_raw (см)', color=colors['raw'], alpha=0.5)
    ax.plot(df['frame'], df['E_fil_cm'], label='E_fil (см)', color=colors['filtered'])

    # Скользящее среднее
    roll_raw = df['E_raw_cm'].rolling(window_size, center=True).mean()
    roll_fil = df['E_fil_cm'].rolling(window_size, center=True).mean()

    ax.plot(df['frame'], roll_raw, '--', color=colors['mean'], label='Mean_raw')
    ax.plot(df['frame'], roll_fil, '--', color=colors['steady'], label='Mean_fil')

    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Кадр')
    ax.set_ylabel('Ошибка (см)')
    ax.set_title('Ошибка следования за линией')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_cm.png', dpi=300)
    plt.close()

    print("✓ error_cm.png сохранён")

    # ===================================
    #   2. Установившаяся ошибка (EMA)
    # ===================================
    steady = df['error_filtered'].ewm(alpha=0.05).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['frame'], df['error_filtered'], label='E_fil (px)', color=colors['filtered'], alpha=0.6)
    ax.plot(df['frame'], steady, label='EMA', color=colors['steady'], linewidth=2)

    final_ss = steady.iloc[-1]
    ax.axhline(final_ss, linestyle=':', color=colors['steady'], label=f'Final={final_ss:.2f}')

    ax.set_xlabel('Кадр')
    ax.set_ylabel('Ошибка (px)')
    ax.set_title('Установившаяся ошибка (EMA)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'steady_state.png', dpi=300)
    plt.close()

    print("✓ steady_state.png сохранён")

    # # ====================
    # #   3. Угол линии
    # # ====================
    # fig, ax = plt.subplots(figsize=(14, 6))

    # ax.plot(df['frame'], df['angle_deg'], label='Angle', color='#e67e22')
    # roll_angle = df['angle_deg'].rolling(window_size, center=True).mean()
    # ax.plot(df['frame'], roll_angle, '--', color='#d35400', label='Mean')

    # ax.axhline(0, color='gray', linewidth=0.8)
    # ax.set_xlabel('Кадр')
    # ax.set_ylabel('Угол (град)')
    # ax.set_title('Угол линии')
    # ax.legend()
    # ax.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig(output_dir / 'angle.png', dpi=300)
    # plt.close()

    # print("✓ angle.png сохранён")

    # # ====================
    # #   4. Скорость
    # # ====================
    # fig, ax = plt.subplots(figsize=(14, 6))

    # ax.plot(df['frame'], df['v'], label='V', color='#16a085')
    # roll_v = df['v'].rolling(window_size, center=True).mean()
    # ax.plot(df['frame'], roll_v, '--', color='#0e6655', label='Mean')

    # ax.set_xlabel('Кадр')
    # ax.set_ylabel('Скорость (PWM)')
    # ax.set_title('Скорость робота')
    # ax.legend()
    # ax.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig(output_dir / 'velocity.png', dpi=300)
    # plt.close()

    # print("✓ velocity.png сохранён")

    # # ====================
    # #   5. Буст скорости
    # # ====================
    # fig, ax = plt.subplots(figsize=(14, 6))
    # ax.plot(df['frame'], df['straight_boost'], label='Boost', color='#f39c12')
    # ax.fill_between(df['frame'], 0, df['straight_boost'], alpha=0.2, color='#f39c12')

    # roll_boost = df['straight_boost'].rolling(window_size, center=True).mean()
    # ax.plot(df['frame'], roll_boost, '--', label='Mean', color='#e67e22')

    # ax.set_xlabel('Кадр')
    # ax.set_ylabel('Буст (PWM)')
    # ax.set_title('Буст скорости на прямых участках')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(output_dir / 'boost.png', dpi=300)
    # plt.close()

    # print("✓ boost.png сохранён")

    # # ====================
    # #   6. Команды моторам
    # # ====================
    # fig, ax = plt.subplots(figsize=(14, 6))

    # ax.plot(df['frame'], df['left_cmd'], label='Left', color='#c0392b', alpha=0.5)
    # ax.plot(df['frame'], df['right_cmd'], label='Right', color='#2980b9', alpha=0.5)

    # roll_l = df['left_cmd'].rolling(window_size, center=True).mean()
    # roll_r = df['right_cmd'].rolling(window_size, center=True).mean()

    # ax.plot(df['frame'], roll_l, '--', label='L_mean', color='#a93226')
    # ax.plot(df['frame'], roll_r, '--', label='R_mean', color='#1f618d')

    # ax.set_xlabel('Кадр')
    # ax.set_ylabel('PWM')
    # ax.set_title('Команды моторам')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(output_dir / 'motor_cmds.png', dpi=300)
    # plt.close()

    # print("✓ motor_cmds.png сохранён")

    # # ====================
    # #   7. FPS
    # # ====================
    # fig, ax = plt.subplots(figsize=(14, 6))
    # ax.plot(df['frame'], df['fps'], label='FPS', color='#8e44ad')

    # roll_fps = df['fps'].rolling(window_size, center=True).mean()
    # ax.plot(df['frame'], roll_fps, '--', label='Mean', color='#6c3483')

    # ax.set_xlabel('Кадр')
    # ax.set_ylabel('FPS')
    # ax.set_title('Производительность (FPS)')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(output_dir / 'fps.png', dpi=300)
    # plt.close()

    # print("✓ fps.png сохранён")

    # # ============================
    # #   8. Направления (гистограмма)
    # # ============================
    # direction_counts = df['direction'].value_counts()

    # fig, ax = plt.subplots(figsize=(10, 6))
    # bars = ax.bar(direction_counts.index, direction_counts.values,
    #               color=['#3498db', '#2ecc71', '#e74c3c'])

    # for bar in bars:
    #     h = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width() / 2, h,
    #             f'{int(h)}\n({h/len(df)*100:.1f}%)',
    #             ha='center', va='bottom')

    # ax.set_xlabel('Направление')
    # ax.set_ylabel('Кол-во кадров')
    # ax.set_title('Распределение направлений')
    # plt.tight_layout()
    # plt.savefig(output_dir / 'direction_hist.png', dpi=300)
    # plt.close()

    # print("✓ direction_hist.png сохранён")

#     # ================================
#     #   9. Сводный дашборд
#     # ================================
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     fig.suptitle('Телеметрия робота', fontsize=16)

#     # Ошибка
#     axes[0, 0].plot(df['frame'], df['E_raw_cm'], 'r-', alpha=0.3, label='E_raw')
#     axes[0, 0].plot(df['frame'], roll_fil, 'b-', label='E_fil')
#     axes[0, 0].set_title('Ошибка (см)')
#     axes[0, 0].legend()
#     axes[0, 0].grid(True)

#     # Моторы
#     axes[0, 1].plot(df['frame'], df['left_cmd'], 'r-', label='L')
#     axes[0, 1].plot(df['frame'], df['right_cmd'], 'b-', label='R')
#     axes[0, 1].set_title('Команды моторам')
#     axes[0, 1].legend()
#     axes[0, 1].grid(True)

#     # Скорость + угол

#     # 9.3 Скорость + Угол
#     ax1 = axes[1, 0]
#     ax2 = ax1.twinx()
    
#     line1 = ax1.plot(df['frame'], df['v'], 'g-', linewidth=1.5, label='Velocity')
#     line2 = ax2.plot(df['frame'], df['angle_deg'], 'orange', linewidth=1.5, label='Angle')
    
#     ax1.set_xlabel('Кадр')
#     ax1.set_ylabel('Скорость (PWM)', color='g')
#     ax2.set_ylabel('Угол fitline (deg)', color='orange')
#     ax1.tick_params(axis='y', labelcolor='g')
#     ax2.tick_params(axis='y', labelcolor='orange')
    
#     lines = line1 + line2
#     labels = [l.get_label() for l in lines]
#     ax1.legend(lines, labels, loc='upper left')
#     ax1.set_title('Скорость и угол', fontweight='bold')
#     ax1.grid(True, alpha=0.3)

#     # Таблица статистики
#     axes[1, 1].axis('off')
#     stats = [
#         ["Метрика", 'Среднее', 'Дисперсия', 'Мин', 'Макс'],
#         ['Ошибка положения линии (cm)', f'{roll_fil.mean():.2f}',
#                        f'{roll_fil.std():.2f}',
#                        f'{roll_fil.min():.2f}',
#                        f'{roll_fil.max():.2f}'],
#         ['Угол', f'{df["angle_deg"].mean():.1f}',
#                   f'{df["angle_deg"].std():.1f}',
#                   f'{df["angle_deg"].min():.1f}',
#                   f'{df["angle_deg"].max():.1f}'],
#         ['Скорость pwm', f'{df["v"].mean():.1f}',
#               f'{df["v"].std():.1f}',
#               f'{df["v"].min():.1f}',
#               f'{df["v"].max():.1f}'],
#         ['Частота кадов', f'{df["fps"].mean():.1f}',
#                 f'{df["fps"].std():.1f}',
#                 f'{df["fps"].min():.1f}',
#                 f'{df["fps"].max():.1f}'],
#     ]

#     table = axes[1, 1].table(cellText=stats, cellLoc='center', loc='center',
#                              colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1, 2)
#     for i in range(5):
#         table[(0, i)].set_facecolor('#3498db')
#         table[(0, i)].set_text_props(weight='bold', color='white')
#    # axes[1, 1].set_title('Общая статистика', fontweight='bold', pad=20)
    
#     plt.tight_layout()
#     plt.savefig(output_dir / 'dashboard.png', dpi=300)
#     plt.close()

#     print("✓ dashboard.png сохранён")


    # === 9. СВОДНАЯ СТАТИСТИКА === #
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Большой заголовок
    fig.suptitle('Summary Dashboard', fontsize=26, fontweight='bold', y=0.995)

    # Глобальный размер шрифта
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })

    # 9.1 Ошибка
    axes[0, 0].plot(df['frame'], df['E_raw_cm'], 'r-', alpha=0.3, linewidth=1.8, label='Raw')
    axes[0, 0].plot(df['frame'], roll_fil, 'b-', linewidth=2.2, label='Filtered')
    axes[0, 0].axhline(0, color='gray', linestyle='--', linewidth=1)

    axes[0, 0].set_title('Error (cm)', fontweight='bold', fontsize=20)
    axes[0, 0].set_xlabel('Frame', fontsize=17)
    axes[0, 0].set_ylabel('Error (cm)', fontsize=17)
    axes[0, 0].legend(fontsize=15)
    axes[0, 0].grid(True, alpha=0.3)

    # 9.2 Motor Commands
    axes[0, 1].plot(df['frame'], df['left_cmd'], 'r-', linewidth=1.8, label='Left')
    axes[0, 1].plot(df['frame'], df['right_cmd'], 'b-', linewidth=1.8, label='Right')

    axes[0, 1].set_title('Motor Commands', fontweight='bold', fontsize=20)
    axes[0, 1].set_xlabel('Frame', fontsize=17)
    axes[0, 1].set_ylabel('PWM', fontsize=17)
    axes[0, 1].legend(fontsize=15)
    axes[0, 1].grid(True, alpha=0.3)

    # 9.3 Скорость + Угол
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()

    line1 = ax1.plot(df['frame'], df['v'], 'g-', linewidth=1.8, label='Velocity')
    line2 = ax2.plot(df['frame'], df['angle_deg'], 'orange', linewidth=1.8, label='Angle')

    ax1.set_xlabel('Frame', fontsize=17)
    ax1.set_ylabel('Velocity (PWM)', color='g', fontsize=17)
    ax2.set_ylabel('Angle Fitline (deg)', color='orange', fontsize=17)

    ax1.tick_params(axis='y', labelcolor='g', labelsize=14)
    ax2.tick_params(axis='y', labelcolor='orange', labelsize=14)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=15)

    ax1.set_title('Velocity & Angle', fontweight='bold', fontsize=20)
    ax1.grid(True, alpha=0.3)

    # 9.4 Таблица статистики
    axes[1, 1].axis('off')

    stats_data = [
        ['Metric', 'Mean', 'Std', 'Min', 'Max'],
        ['Error (filtered)', f'{roll_fil.mean():.2f}',
                            f'{roll_fil.std():.2f}',
                            f'{roll_fil.min():.2f}',
                            f'{roll_fil.max():.2f}'],
        ['Angle Fitline (deg)', f'{df["angle_deg"].mean():.1f}',
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

    # Увеличенная таблица
    table = axes[1, 1].table(
        cellText=stats_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.28, 0.17, 0.17, 0.17, 0.17]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(16)       # ← Шрифт таблицы увеличен
    table.scale(1.4, 2.2)        # ← Высота увеличена

    # Заголовки таблицы
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=18)

    axes[1, 1].set_title('Statistics Summary', fontweight='bold', fontsize=20, pad=25)

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: summary_dashboard.png")


if __name__ == "__main__":
    csv_path = "/home/angelika/Desktop/Seoul/Vision-Based-Line-Following-Car/data/for_analysis/linetracer_thresh_20251119_153730_labels.txt"
    visualize_telemetry(csv_path, output_dir="./telemetry_plots", max_frames=520, window_size=20)
