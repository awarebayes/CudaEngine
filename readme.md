# CudaGL

**Цель проекта:** Создание программного растеризатора, работающего достаточно быстро на современной видеокарте

credits: [tinyrenderer](https://github.com/ssloy/tinyrenderer), [learnopengl](https://learnopengl.com/)

## Трехмерная графика и зачем нужна видеокарта

Обычный draw call в программном растеризаторе выглядит примерно так:

```python
for model in models:
    for face in model.vertex:
        face = transform @ face
        triangle(face)
        
def triangle(face):
    bbox = find_bbox(face)
    for x in range(bbox.x_min, bbox.x_max):
        for y in range(bbox.y_min, bbox.y_max):
            bc = barycentric(face, x, y)
            z = interpolate(face, bc)
            if z < z_buffer[x][y]:
                z_buffer[x][y] = z
                draw_pixel(x, y, face, bc)

def draw_pixel(x, y, face, bc):
    texture_uv = interpolate_texure(face, bc)
    normal_uv = interpolate_noramal(face, bc)
    texture = texture_map.get_uv(texture_uv)
    normal = normal_map.get_uv(normal_uv)
    color = texture.color * dot(normal.color, light_dir)
    # ...
    # other boring shading code
    put_pixel(x, y, color)
```

Этот код можно параллелизовать в нескольких местах.

Допустим, его можно параллелизовать два внешних цикла, в shader_triangle мы проверяем что 
z пиксела равен z буффера.

```python
for model in parlallel(models):
    for face in parallel(model.vertex):
        face = transform @ face
        z_buffer_triangle(face)
        
sync_barrier()
        
for model in parlallel(models):
    for face in parallel(model.vertex):
        face = transform @ face
        shader_triangle(face)
```

## Выбор языка и описание примитивов

Для написания проекта использован [Nvidia Cuda](https://developer.nvidia.com/cuda-toolkit).

Критерием выбора языка для проекта является:
- быстрота выполнения программы
- возможность простой многопоточности на 
уровне задач, а не на уровне тредов.
- хорошая математическая библиотека из коробки
- тот факт, что я иногда пишу на нем на работе

## Введение

Большинство программ написаны для выполнения на процессоре. Такой стиль программирования не подойдет для написания 
высокопроизводительного рендера. Процессор не предназначен для мультитрединга с большим числом потоков, т.к. у него
их попросту недостаточное кол-во. У моей видеокарты (gtx 1080) 1920 Cuda-ядер, а у моего процессора (Ryzen 7 3800x) всего 32.

![](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/gpu-devotes-more-transistors-to-data-processing.png)

С этим приходит и недостаток, видеокарта имеет архитектуру SIMT - Single Instruction Multiple Threads. 

Видеокарта делится на сетки, внутри сетки блок. Каждый блок имеет thread schelduler.

Одна инструкция выполняется для целого блока потоков. Блок в свою очередь делится на варпы по 32 треда в каждом.

Потоки в варпе, которым не адресована инструкция, простаивают.

Это контрастирует с процессором, где каждый поток выполняет свои инструкцию.

![](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/grid-of-thread-blocks.png)

Следовательно, если вставить if внутри программы, выполняющейся на потоке,
половина тредов может выполнять первый branch if'а, а дальше ждать пока вторая половина выполнит второй branch.
Программа будет работать в два раза медленнее из-за if'а.