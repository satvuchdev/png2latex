import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from scipy.interpolate import PchipInterpolator
import subprocess











# ======================================================= НАСТРОЙКИ =======================================================
path_to_folder_with_this_file = os.path.dirname(os.path.abspath(__file__))
print(f"Текущая рабочая директория: {path_to_folder_with_this_file}")


OUTPUT_LATEX = f"curve_equations_latex.txt"
RDP_EPSILON = int(input("Введите коэф. упрощения для алгоритма Рамера–Дугласа–Пекера (Наилучший результат при E = 5): "))
MIN_POINTS_AFTER_RDP = 1                                      # Минимальное кол-во точек после RDP, если контур слишком мал
# =========================================================================================================================






















# =========================== Алгоритм Рамера–Дугласа–Пекера ===========================
def rdp(points, eps):                                                                                                 # РАЗДЕЛ ОТЧЕТА - 8
    '''
    Алгоритм Рамера-Дугласа-Пекера (RDP) – алгоритм упрощения ломаных линий
    путем удаления точек, не оказывающих существенного влияния на форму контура.
    Позволяет уменьшить количество точек перед интерполяцией.
    '''


    if len(points) < 3:
        return points
    start = np.array(points[0])                                                                                       # Начальная точка
    end = np.array(points[-1])                                                                                        # Конечная точка
    seg = end - start                                                                                                 # Вектор сегмента
    seg_len2 = seg.dot(seg)                                                                                           # Квадрат длины сегмента
    if seg_len2 == 0:                                                                                                 # Если начальная и конечная точки совпадают
        dists = np.linalg.norm(np.array(points) - start, axis=1)                                                      # Расстояния от всех точек до начальной точки
    else:
        t = np.dot(np.array(points) - start, seg) / seg_len2                                                          # Параметризация точек вдоль сегмента
        proj = np.outer(t, seg) + start                                                                               # Проекции точек на сегмент
        dists = np.linalg.norm(np.array(points) - proj, axis=1)                                                       # Расстояния от точек до проекций на сегмент
    idx = np.argmax(dists)                                                                                            # Индекс точки с максимальным расстоянием
    if dists[idx] <= eps:                                                                                             # Если максимальное расстояние меньше порога
        return [tuple(start), tuple(end)]                                                                             # Удаление всех промежуточных точек
    else:
        left = rdp(points[:idx+1], eps)                                                                               # Рекурсивный вызов для левой части
        right = rdp(points[idx:], eps)                                                                                # Рекурсивный вызов для правой части
        return left[:-1] + right                                                                                      # Объединение результатов, избегая дублирования точки


def plot_rdp_points(contours):                                                                                        # РАЗДЕЛ ОТЧЕТА - 8
    plt.figure(figsize=(8, 8))
    for contour in contours:                                                                                          # Отображение исходного контура
        arr, _ = process_contour(contour)
        plt.scatter(arr[:, 0], arr[:, 1],
                    s=40,
                    c="red",
                    zorder=3)                                                                                         # Характерные точки после RDP
    plt.title("Характерные точки\nпосле применения алгоритма Рамера–Дугласа–Пекера")
    plt.show()
# =====================================================================================







# ========================== Предварительная обработка в RGBA ==========================
class GetPng:                                                                                                         # РАЗДЕЛ ОТЧЕТА - 4
    def __init__(self, file):
        self.file = file


    def open(self):
        '''
        RGBA – модель представления цвета пикселя, включающая красный (R), зеленый (G),
        синий (B) и альфа-канал (A), отвечающий за прозрачность.
        Используется при анализе пикселей изображения.
        '''


        if not os.path.isfile(self.file):
            raise FileNotFoundError(f"Файл не найден: {self.file}")
        return Image.open(self.file).convert("RGBA")
# =====================================================================================







# ======================== Построение бинарной маски и контуров ========================
class ChoosePoints:                                                                                                   # РАЗДЕЛ ОТЧЕТА - 5
    def __init__(self, image):
        self.image = image
        self.mask = None

    def is_white(self, px):
        r, g, b, a = px
        return (r > 240 and g > 240 and b > 240 and a > 200)                                                          # Определение белого цвета с учётом прозрачности

    def build_mask(self):
        '''
        Бинарная маска – частный случай бинарного изображения, который выполняет функцию логического фильтра,
        указывая, какие области исходного изображения подлежат обработке, а какие игнорируются.
        '''


        w, h = self.image.size                                                                                         # Получение размеров изображения
        px = self.image.load()                                                                                         # Загрузка пикселей изображения
        self.mask = np.zeros((h, w), dtype=np.uint8)                                                                   # Инициализация бинмаски с нулями
        for y in range(h):                                                                                             # Проход по всем пикселям изображения
            for x in range(w):
                if self.is_white(px[x, y]):                                                                            # Если пиксель белый, устанавливаем значение маски в 1
                    self.mask[y, x] = 1
        return self.mask

    def detect_contours(self):
        if self.mask is None:                                                                                          # Построение маски, если она ещё не создана
            self.build_mask()
        contours = find_contours(self.mask, level=0.5)                                                                 # Поиск контуров в бинарной маске
        return contours
# =====================================================================================







# ============================ Обработка и упрощение контура ============================
def process_contour(contour, rdp_eps=RDP_EPSILON):                                                                      # РАЗДЕЛ ОТЧЕТА - 9
    '''
    Преобразование контура в массив точек и параметризацию по длине дуги.
    '''
    pts = [(float(p[1]), float(p[0])) for p in contour]                                                                 # Преобразование координат (row, col) в (x, y)
    if np.linalg.norm(np.array(pts[0]) - np.array(pts[-1])) > 1e-5:                                                     # Замыкание контура, если начальная и конечная точки не совпадают (1e-5 - малое значение для сравнения)
        pts.append(pts[0])                                                                                              # Начальная точка становится конечной
    pts_reduced = rdp(pts, rdp_eps)                                                                                     # Упрощение контура с помощью RDP
    if np.linalg.norm(np.array(pts_reduced[0]) - np.array(pts_reduced[-1])) > 1e-5:                                     # Замыкание упрощённого контура
        pts_reduced.append(pts_reduced[0])                                                                              # Начальная точка становится конечной
    if len(pts_reduced) < MIN_POINTS_AFTER_RDP:                                                                         # Если после RDP слишком мало точек, оставляем исходные точки с равномерным шагом
        pts_reduced = pts[::max(1, len(pts)//MIN_POINTS_AFTER_RDP)]                                                     # Расчет равномерного шага
        if pts_reduced[0] != pts_reduced[-1]:                                                                           # Замыкание контура
            pts_reduced.append(pts_reduced[0])                                                                          # Начальная точка становится конечной
    arr = np.array(pts_reduced)                                                                                         # Преобразование в numpy массив
    diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)                                                                # Вычисление длин дуг между последовательными точками
    cum = np.concatenate([[0.0], np.cumsum(diffs)])                                                                     # Кумулятивная длина дуг - расстояние от начальной точки                                        
    total = cum[-1] if cum[-1] > 0 else 1.0                                                                             # Предотвращение деления на ноль
    t = cum / total                                                                                                     # Параметризация по длине дуги [0, 1]       
    return arr, t
# =======================================================================================







# ========================= Перевод в многочлен третьей степени =========================
def expand_shifted_poly(coefs_u, t0):                                                                                   # РАЗДЕЛ ОТЧЕТА - 10
    a, b, c, d = coefs_u                                                                                                # Коэффициенты многочлена третьей степени в форме сдвинутого базиса (t - t0)
    A3 = a                                                                                                              # Коэффициент при t^3 остаётся неизменным
    A2 = b - 3*a*t0                                                                                                     # Коэффициент при t^2
    A1 = c - 2*b*t0 + 3*a*t0**2                                                                                         # Коэффициент при t^1
    A0 = d - c*t0 + b*t0**2 - a*t0**3                                                                                   # Свободный член
    return np.array([A3, A2, A1, A0], dtype=float)                                                                      # Возврат коэффициентов в стандартной форме
# =======================================================================================







# ========================= Аппроксимация кусочно-кубическим сплайном =========================
def fit_piecewise_cubic(arr, t):                                                                                         # РАЗДЕЛ ОТЧЕТА - 11
    x, y = arr[:, 0], arr[:, 1]                                                                                          # Разделение координат на x и y для последующей интерполяции
    cs_x = PchipInterpolator(t, x)                                                                                       # Создание PCHIP интерполятора для x
    cs_y = PchipInterpolator(t, y)                                                                                       # Создание PCHIP интерполятора для y
    nseg = len(t) - 1                                                                                                    # Количество сегментов сплайна (-1, так как t содержит n+1 точек для n сегментов)
    segments = []                                                                                                        # Список для хранения коэффициентов каждого сегмента
    for i in range(nseg):                                                                                                # Проход по каждому сегменту
        t0, t1 = t[i], t[i+1]                                                                                            # Границы текущего сегмента
        coefs_x_local = cs_x.c[:, i].astype(float)                                                                       # Коэффициенты многочлена для x в текущем сегменте
        coefs_y_local = cs_y.c[:, i].astype(float)                                                                       # Коэффициенты многочлена для y в текущем сегменте
        coefs_x_expanded = expand_shifted_poly(coefs_x_local, t0)                                                        # Преобразование коэффициентов по х в стандартную форму
        coefs_y_expanded = expand_shifted_poly(coefs_y_local, t0)                                                        # Преобразование коэффициентов по у в стандартную форму
        segments.append({                                                                                                # Сохранение информации о сегменте в список
            "interval": (float(t0), float(t1)),
            "coefs_x": coefs_x_expanded,
            "coefs_y": coefs_y_expanded,
        })
    return segments
# ==============================================================================================







# =================================== Запись в LaTeX формат ====================================
def write_latex(all_segments, filename=OUTPUT_LATEX):                                                                   # РАЗДЕЛ ОТЧЕТА - 13
    lines = []
    for curve_i, segments in enumerate(all_segments):                                                                   # Проход по всем кривым
        lines.append(f"=== Кривая {curve_i+1} ===\n")                                                                   # Заголовок для каждой кривой для последующей записи в латех
        for seg_i, seg in enumerate(segments):                                                                          # Проход по сегментам текущей кривой
            t0, t1 = seg["interval"]                                                                                    # Интервал параметра t для текущего сегмента (t - это параметр длины дуги, отвечающий за положение на кривой)
            ax, bx, cx, dx = seg["coefs_x"]                                                                             # Коэффициенты многочлена x(t)
            ay, by, cy, dy = seg["coefs_y"]                                                                             # Коэффициенты многочлена y(t)
            lines.append(f"Сегмент {seg_i+1} ($t \\in [{t0:.6f}, {t1:.6f}]$):\n")                                       # Заголовок для каждого сегмента с указанием интервала t
            lines.append(f"$$x(t) = {ax:.12g} t^3 + {bx:.12g} t^2 + {cx:.12g} t + {dx:.12g}$$")                         # Запись уравнения x(t) в формате LaTeX
            lines.append(f"$$y(t) = {ay:.12g} t^3 + {by:.12g} t^2 + {cy:.12g} t + {dy:.12g}$$\n")                       # Запись уравнения y(t) в формате LaTeX
        lines.append("\n")
    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"LaTeX файл сохранён: {filename}")
# ============================================================================================







# ================================== Визуализация сплайнов ====================================
def plot_splines(contours, all_segments):
    plt.figure(figsize=(8, 8))
    for contour in contours:                                                                                            # Отображение исходного контура
        xy = np.array([(p[1], p[0]) for p in contour])                                                                  # Преобразование координат (row, col) в (x, y)
        plt.plot(xy[:, 0], xy[:, 1], linestyle='--', alpha=0.5)                                                         # Отрисовка исходного контура пунктиром
    for segments in all_segments:                                                                                       # Отображение аппроксимированных кривых
        for seg in segments:                                                                                            # Проход по сегментам текущей кривой
            t0, t1 = seg["interval"]                                                                                    # Интервал параметра t для текущего сегмента
            ax, bx, cx, dx = seg["coefs_x"]                                                                             # Коэффициенты многочлена x(t)
            ay, by, cy, dy = seg["coefs_y"]                                                                             # Коэффициенты многочлена y(t)
            ts = np.linspace(t0, t1, 40)                                                                                # Генерация значений t в интервале сегмента
            xs = ax*ts**3 + bx*ts**2 + cx*ts + dx                                                                       # Вычисление x(t) для каждого t
            ys = ay*ts**3 + by*ts**2 + cy*ts + dy                                                                       # Вычисление y(t) для каждого t
            plt.plot(xs, ys, linewidth=2)                                                                               # Отрисовка сегмента сплайна
    plt.title("Аппроксимированные сплайны\nв сравнении с исходными контурами бинарной маски")
    plt.show()
# ==============================================================================================







# ====================================== Бинарная маска ========================================
def show_mask(mask):                                                                                                    # РАЗДЕЛ ОТЧЕТА - 5
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray")                                                                                       # Отображение бинарной маски, gray - тип маски
    plt.title("Бинарная маска")
    plt.axis('off')                                                                                                     # Отключение осей для чистого отображения
    plt.show()
# ==============================================================================================







# ======================================= Главная функция =======================================
def main():
    image_name = input("Введите имя файла PNG (без .png): ")
    INPUT_PNG = rf"{path_to_folder_with_this_file}\images\{image_name}.png"
    img = GetPng(INPUT_PNG).open()



    w, h = img.size                                                                                                     # Получение размеров изображения
    chooser = ChoosePoints(img)                                                                                         # Инициализация выбора точек
    chooser.build_mask()                                                                                                # Построение бинарной маски
    contours = chooser.detect_contours()                                                                                # Обнаружение контуров в маске
    print(f"Найдено замкнутых контуров объекта: {len(contours)}")                                                       
    all_segments = []                                                                                                   # Список для хранения всех сегментов всех кривых
    good_contours = []                                                                                                  # Список для хранения контуров, прошедших обработку
    for c in contours:                                                                                                  # Обработка каждого контура
        arr, t = process_contour(c)                                                                                     # Преобразование контура в массив точек и параметризацию по длине дуги
        if len(arr) < 4:                                                                                                # Пропуск контуров с недостаточным количеством точек
            continue
        segments = fit_piecewise_cubic(arr, t)                                                                          # Аппроксимация кусочно-кубическим сплайном
        all_segments.append(segments)                                                                                   # Добавление сегментов текущего контура в общий список
        good_contours.append(c)                                                                                         # Добавление обработанного контура в список хороших контуров
# =============================================================================================



















                                                                                                             # ПРИМЕР - РАЗДЕЛ ОТЧЕТА - 19
# =================================================== Поэтапная визуализация результатов ==================================================
    with open(INPUT_PNG, "rb") as png_image:                                                                 # Открытие изображения             (РАЗДЕЛ ОТЧЕТА - 4)
        subprocess.run(["start", "explorer", "/open,", INPUT_PNG], shell=True) 

    mask = chooser.build_mask()                                                                              # Бинарная маска                   (РАЗДЕЛЫ ОТЧЕТА - 5, 6, 7)
    show_mask(mask)
    
    plot_rdp_points(good_contours)                                                                           # Характерные точки после RDP      (РАЗДЕЛЫ ОТЧЕТА - 8, 16)

    plot_splines(good_contours, all_segments)                                                                # Аппроксимированные сплайны       (РАЗДЕЛЫ ОТЧЕТА - 9, 10, 11, 14, 17)

    write_latex(all_segments)                                                                                # Запись в LaTeX формат            (РАЗДЕЛЫ ОТЧЕТА- 13.1, 13.2)
# ========================================================================================================================================



















if __name__ == "__main__":
    main()