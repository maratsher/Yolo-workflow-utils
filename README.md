# Darknet utilities
**Darknet utilities** - это репозиторий, содержащий в себе скрипты для работы с фреймворком глубокого обучения [Darknet](https://github.com/AlexeyAB/darknet).

### Основные компоненты

| библиотека | версия |
| ------ | ------ |
| tqdm | 4.36.1 |
| opencv-python | 3.4.0.14 |
| numpy | 1.17.2 |
| imgaug | 0.3.0 |

---

### Краткое описание:

   - [check_images_for_label](#check_images_for_label) - проверка наличия файла разметки к изображению.
   - [check_warning_boxes](#check_warning_boxes) - проверка изображений на проблемные boxes.
   - [remove_classes](#remove_classes) - удаление не нужных классов из разметки исходя из файла classes.
   - [remove_image_in_txt](#remove_image_in_txt) - удаление путей из текстового файла в соответствии с bad_list.
   - [remove_badlist](#remove_badlist) - удаление данных из bad list.
   - [split_dataset_into_samples_detector](#split_dataset_into_samples_detector) - формирование обучающей, тестовой (валидационной) выборки из датасета для обучения детектора.
   - [split_dataset_into_samples_classifier](#split_dataset_into_samples_classifier) - формирование обучающей, тестовой (валидационной) выборки из датасета для обучения классификатора.
   - [augmentation_dataset](#augmentation_dataset) - формирование новых данных посредством аугментации.
   - [models_testing](#models_testing) - тестирование yolo-моделей.


## Скрипты

### check_images_for_label
**Описание:** *Скрипт проверяет наличие файлов разметки к изображениям в датасете. Если разметка к файлу отсутствует, то путь до изображения записывается в bad.list после чего текстовый файл со всеми проблемными изображениями сохраняется.*

**Параметры:**

|  |  |  |
| ------ | ------ | ------ |
| -d | --dir_to_dataset | Путь до датасета |


**Использование:**

> python3 check_images_for_label.py -d {%path_to_dataset%}
> 
> python3 check_images_for_label.py -d /media/dkubatin/39e992da-a01a-4384-a580-e798bb2aab2a/datasets

### check_warning_boxes
**Описание:** *Скрипт проверяет файлы разметки на проблемные bounding boxes. Если таковые есть, то создается файл bad_list_boxes.txt. Есть параметр, с помощью которого данные bboxes можно удалить из разметки.*

**Параметры:**

|  |  |  |
| ------ | ------ | ------ |
| -d | --dir_to_dataset | Путь до датасета |
| -r | --remove_boxes | Флаг, удалить проблемные bboxes из разметки |


**Использование:**

> python3 check_warning_boxes.py -d {%path_to_dataset%} -r
> 
> python3 check_warning_boxes.py -d /media/dkubatin/39e992da-a01a-4384-a580-e798bb2aab2a/datasets -r

---

### remove_classes
**Описание:** *Скрипт проверяет классы в разметочных файлах. Если номер класса больше чем конечное число классов, то данный bbox удаляется из разметки. Если файл разметки пустой, то изображений и файла разметки переносятся в созданную папку negative.*

**Параметры:**

|  |  |  |
| ------ | ------ | ------ |
| -d | --dir_to_dataset | Путь до датасета |
| -с | --classes | Путь до файла с классами |


**Использование:**

> python3 remove_classes.py -d {%path_to_dataset%} -c {%path_to_file_classes%}
> 
> python3 remove_classes.py -d /media/dkubatin/39e992da-a01a-4384-a580-e798bb2aab2a/datasets -c /media/dkubatin/39e992da-a01a-4384-a580-e798bb2aab2a/classes.txt

---

### remove_image_in_txt
**Описание:** *Скрипт удаляет из текстового файла (train.txt, test.txt, valid.txt) пути до изображений исходя из файла bad_list.*

**Параметры:**

|  |  |  |
| ------ | ------ | ------ |
| -b | --bad_list | Путь до bad листа |
| -t | --text_file | Путь до текстового файла |


**Использование:**

> python3 remove_image_in_txt.py -b {%path_to_badlist%} -t {%path_to_text_file%}
> 
> python3 remove_image_in_txt.py -b /home/dkubatin/PycharmProjects/darknet_utilities/bad.list -t /home/dkubatin/2/train.txt

---

### remove_badlist
**Описание:** *Скрипт удаляет файлы в соответствии с данными из bad list.*

**Параметры:**

|  |  |  |
| ------ | ------ | ------ |
| -b | --bad_list | Путь до bad листа |


**Использование:**

> python3 remove_badlist.py -b {%path_to_bad_list%}
> 
> python3 remove_badlist.py -b /home/dkubatin/PycharmProjects/darknet_utilities/bad.list

---

### split_dataset_into_samples_detector
**Описание:** *Скрипт формирует (обучающую - train, тестовую -test, валидационную - validation) исходя из датасета для обучения детектора. На вход подается путь до датасета, путь до сохранения выборок, процентное соотношение выборок и параметр рандомизации данных (формирование выборок в случайном порядке)*

**Параметры:**

|  |  |  |
| ------ | ------ | ------ |
| -d | --dir_to_dataset | Путь до датасета |
| -s | --save_path | Путь сохранения выборок |
| -sr | --sample_ratio | Процентное соотношение выборок |
| -r | --random | Флаг, данные в случайном порядке |


**Использование:**

> python3 split_dataset_into_samples_detector.py -d {%path_to_dataset%} -s {%save_to_samples%} -sr {%sample_ratio%} -r
> 
> python3 split_dataset_into_samples_detector.py -d /media/dkubatin/39e992da-a01a-4384-a580-e798bb2aab2a/datasets -s /media/dkubatin/39e992da-a01a-4384-a580-e798bb2aab2a/data_learn_sample1 -sr 80_20 -r

---


### split_dataset_into_samples_classifier
**Описание:** *Скрипт формирует (обучающую - train, тестовую -test, валидационную - validation) исходя из датасета для обучения классификатора. На вход подается путь до датасета, путь до сохранения выборок, процентное соотношение выборок и параметр рандомизации данных (формирование выборок в случайном порядке)*

**Параметры:**

|  |  |  |
| ------ | ------ | ------ |
| -d | --dir_to_dataset | Путь до датасета |
| -s | --save_path | Путь сохранения выборок |
| -sr | --sample_ratio | Процентное соотношение выборок |


**Использование:**

> python3 split_dataset_into_samples_classifier.py -d {%path_to_dataset%} -s {%save_to_samples%} -sr {%sample_ratio%}
> 
> python3 split_dataset_into_samples_classifier.py -d /media/dkubatin/39e992da-a01a-4384-a580-e798bb2aab2a/datasets -s /media/dkubatin/39e992da-a01a-4384-a580-e798bb2aab2a/data_learn_sample1 -sr 80_20

---

### augmentation_dataset
**Описание:** *Скрипт формирует аугментационные данные на основе поданного датасета. На вход подается путь до датасета, путь до сохранения аугментационных данных, и параметры аугментации.*

**Параметры:**

|  |  |  |
| ------ | ------ | ------ |
| -d | --dir_to_dataset | Путь до датасета |
| -s | --save_to_augmentation | Путь сохранения аугмент. данных |
| -mb | --motion_blur | Опция, размытие в движении |
| -dark | --darkness | Опция, затемнение |
| -bright | --brightness | Опция, засветление |
| -noise | --noise | Опция, шум |
| -cont | --contrast | Опция, контрастность (Клахе) |
| -sh | --shadow | Опция, тень на bbox |
| -sh_c | --shadow_class | Перечисление классов, на которые требуется накладывать тень |
| -flip | --flipped | Опция, отображение по горизонтали |
| -random | --random | Флаг, использовать одну из опций аугментации к каждому изображению |


**Использование:**

> python3 augmentation_dataset.py -d {%path_to_dataset%} -s {%save_to_augmentation%} -mb -dark -sh -sh_c 1 6
> 
> python3 augmentation_dataset.py -d /media/dkubatin/39e992da-a01a-4384-a580-e798bb2aab2a/datasets -s /media/dkubatin/39e992da-a01a-4384-a580-e798bb2aab2a/datasets/dataset_aug -mb -dark -sh -sh_c 1 6

---

### models_testing
**Описание:** *Скрипт прогоняет через модель сети тестовые данные, на выход выводится качество работы сети (рассчет основных метрик).*

**Параметры:**

|  |  |  |
| ------ | ------ | ------ |
| -d | --dir_to_dataset | Путь до датасета |
| -c | --cfg_path | Путь до конфига модели сети |
| -dp | --data_path | Путь до файла data |
| -w | --weights_path | Путь до весов сети |
| -l | --letter_box | Опция, использование letter_box |
| -t | --threshold | По-умолчанию: 0.5 |
| -ht | --hier_thresh | По-умолчанию: 0.5 |
| -n | --nms | По-умолчанию: 0.5 |
| -sh | --show_image | Параметр, просмотр N размеченных изображений |
| -gt | --show_ground_true_boxes | Флаг, рисовать ground true разметку |
| -si | --save_image | Флаг, сохранить размеченные изображения |


**Использование:**

В config.yml -> darknet -> libdarknet - указываем путь до файла libdarknet. 

> python3 models_testing.py -d {%dir_to_dataset%} -c {%config_path%} -dp {%data_path%} -w {%weights_path%} -sh 10
> 
> python3 models_testing.py -p /media/dkubatin/39e992da-a01a-4384-a580-e798bb2aab2a/testing_model_plates/testing_datasets/test_video -c /home/dkubatin/test_yolo/yolo-tiny-3l[8049]/yolov3-tiny_3l.cfg -dp /home/dkubatin/test_yolo/yolo-tiny-3l[8049]/plates.data -w /home/dkubatin/test_yolo/yolo-tiny-3l[8049]/yolov3-tiny_3l_best.weights -sh 10